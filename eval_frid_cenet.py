"""FRID computed with CENet features instead of RangeNet++.

Features are the CENet conv_2 activations averaged over valid pixels, so the values are
not on the same scale as eval_frid.py. Results go to frid_cenet_results.txt in the run
directory, or to -o.

Example:
    python eval_frid_cenet.py --run_dir cond_adv_rn_gs6_n1024 --kitti_root $KITTI --kitti_seq 08 --frid_n 1024 --batch_size 8 --also_range ours_rn_s6=cond_adv_rn_gs6_n1024/adv_ranges.pt
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.')
from baseline_fgsm import pcds_to_frid_input, load_kitti_pcds
import attack_guided as ca
from lidm.eval import DATASET_CONFIG
from lidm.eval.eval_utils import compute_fd

_feat = {}
def _hook(m, inp, out):
    _feat['conv_2'] = out.detach()


@torch.no_grad()
def cenet_feats_from_ranges(net, range_b, device):
    x, valid = ca.prepare_for_cenet(range_b.to(device), device)
    ca.net_forward(net, x, 'cenet')
    f = _feat['conv_2']
    v = valid.to(f.dtype)
    num = (f * v).sum(dim=(2, 3))
    den = v.sum(dim=(2, 3)).clamp_min(1.0)
    return (num / den).cpu().numpy()


@torch.no_grad()
def extract(net, range_tensor, device, bs=8):
    feats = []
    for i in range(0, len(range_tensor), bs):
        feats.append(cenet_feats_from_ranges(net, range_tensor[i:i + bs], device))
        torch.cuda.empty_cache()
    return np.vstack(feats)


@torch.no_grad()
def extract_from_pcd_inputs(net, frid_inputs, device, bs=8):
    cfg = dict(DATASET_CONFIG['kitti'])
    dmax = float(cfg['depth_range'][1])
    ranges = []
    for arr in frid_inputs:
        depth_m = arr[0]
        r01 = np.clip(depth_m / dmax, 0.0, 1.0)
        r = r01 * 2.0 - 1.0
        r[depth_m <= 0] = -1.0
        ranges.append(r)
    rt = torch.from_numpy(np.stack(ranges)).unsqueeze(1).float()
    return extract(net, rt, device, bs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--kitti_seq', default='08')
    p.add_argument('--frid_n', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--also_range', nargs='*', default=[])
    p.add_argument('-o', '--output', default=None)
    args = p.parse_args()

    device = torch.device('cuda')
    net = ca.load_cenet(device)
    net.conv_2.register_forward_hook(_hook)
    cfg = dict(DATASET_CONFIG['kitti'])

    print('[frid-cenet] extracting real KITTI features...')
    ref_pcds = load_kitti_pcds(args.kitti_root, args.frid_n, args.kitti_seq)
    ref_feats = extract_from_pcd_inputs(net, pcds_to_frid_input(ref_pcds, cfg), device, args.batch_size)
    print(f'[frid-cenet] real features: {ref_feats.shape}')

    sample_sets = []
    clean_pt = os.path.join(args.run_dir, 'clean_ranges.pt')
    if os.path.exists(clean_pt):
        sample_sets.append(('clean', clean_pt))
    for f in sorted(os.listdir(args.run_dir)):
        if f.startswith('adv_eps') and f.endswith('.pt'):
            sample_sets.append((f.replace('.pt', '').replace('adv_eps', 'attack_eps'),
                                os.path.join(args.run_dir, f)))
    for spec in args.also_range:
        nm, pth = spec.split('=', 1) if '=' in spec else spec.split(':', 1)
        sample_sets.append((nm, pth))

    results = {}
    for name, path in sample_sets:
        print(f'[frid-cenet] {name} ...')
        t = torch.load(path, weights_only=False)[:args.frid_n]
        feats = extract(net, t, device, args.batch_size)
        fd = float(compute_fd(ref_feats, feats))
        results[name] = fd
        print(f'  {name:>30s}: CENet-FRID = {fd:.3f}')
        del t, feats
        torch.cuda.empty_cache()

    out_path = args.output or os.path.join(args.run_dir, 'frid_cenet_results.txt')
    lines = ['=== CENet-FRID vs real KITTI (extractor=CENet conv_2, 128-d) ===',
             f'reference: {args.kitti_root}/sequences/{args.kitti_seq} (n={args.frid_n})',
             'NOTE: not comparable in scale to RangeNet-FRID; ordering within this metric only.', '']
    for k, v in results.items():
        lines.append(f'{k:>30s}: {v:8.3f}')
    text = '\n'.join(lines)
    print('\n' + text)
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'\n[done] saved {out_path}')


if __name__ == '__main__':
    main()