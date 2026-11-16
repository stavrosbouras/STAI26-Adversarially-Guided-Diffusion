"""FRID between real SemanticKITTI scans and sets of generated range images.

FRID is the Frechet distance between RangeNet++ features of real scans and of each set:
clean_ranges.pt and adv_eps*.pt in the run directory, plus any --also_range tensors.
Results go to frid_results.txt in the run directory, or to -o.

Example:
    python eval_frid.py --run_dir fgsm_cn_n1024 --kitti_root $KITTI --kitti_seq 08 --frid_n 1024 --batch_size 4 --also_range ours_s3=cond_adv_cn_n1024/adv_ranges.pt
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.')
from baseline_fgsm import range_to_frid_input, pcds_to_frid_input, load_kitti_pcds
from lidm.eval import build_model, DATASET_CONFIG, AGG_TYPE
from lidm.eval.eval_utils import compute_fd


@torch.no_grad()
def extract_feats_safe(model, inputs_4ch, bs):
    feats = []
    for i in range(0, len(inputs_4ch), bs):
        batch = torch.from_numpy(np.stack(inputs_4ch[i:i + bs])).float().cuda()
        out = model(batch, return_final_logits=True, agg_type=AGG_TYPE)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        feats.append(out)
        del batch
        torch.cuda.empty_cache()
    return np.vstack(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--kitti_seq', default='08')
    p.add_argument('--frid_n', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--also_range', nargs='*', default=[])
    p.add_argument('-o', '--output', default=None)
    args = p.parse_args()

    cfg = dict(DATASET_CONFIG['kitti'])
    feat_model = build_model('kitti', 'rangenet', device='cuda')
    feat_model.eval()

    ref_pcds = load_kitti_pcds(args.kitti_root, args.frid_n, args.kitti_seq)
    ref_feats = extract_feats_safe(feat_model, pcds_to_frid_input(ref_pcds, cfg), args.batch_size)
    print(f'[frid] real features shape: {ref_feats.shape}')

    sample_sets = []
    clean_pt = os.path.join(args.run_dir, 'clean_ranges.pt')
    if os.path.exists(clean_pt):
        sample_sets.append(('clean', clean_pt))
    for f in sorted(os.listdir(args.run_dir)):
        if f.startswith('adv_eps') and f.endswith('.pt'):
            sample_sets.append((f.replace('.pt', '').replace('adv_eps', 'fgsm_eps'),
                                os.path.join(args.run_dir, f)))
    for spec in args.also_range:
        nm, pth = spec.split('=', 1) if '=' in spec else spec.split(':', 1)
        sample_sets.append((nm, pth))

    results = {}
    for name, path in sample_sets:
        t = torch.load(path, weights_only=False)[:args.frid_n]
        feats = extract_feats_safe(feat_model, range_to_frid_input(t, cfg), args.batch_size)
        fd = float(compute_fd(ref_feats, feats))
        results[name] = fd
        print(f'  {name:>30s}: FRID = {fd:.3f}')
        del t, feats
        torch.cuda.empty_cache()

    out_path = args.output or os.path.join(args.run_dir, 'frid_results.txt')
    lines = ['=== FRID vs real KITTI (RangeNet extractor) ===',
             f'reference: {args.kitti_root}/sequences/{args.kitti_seq} (n={args.frid_n})', '']
    for k, v in results.items():
        lines.append(f'{k:>30s}: {v:8.3f}')
    text = '\n'.join(lines)
    print('\n' + text)
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'\n[done] saved {out_path}')


if __name__ == '__main__':
    main()