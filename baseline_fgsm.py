"""FGSM baseline on sem2lidar range images.

For each budget eps, the script applies one FGSM step to the clean range images,
changing only valid (non-empty) pixels, and evaluates CENet and RangeNet++ on the
clean and perturbed images against the conditioning labels. With --from_run it reuses
the clean images and labels of an attack_guided.py run, and unless --skip_frid is set
it also computes FRID against real SemanticKITTI scans. eps is in the [-1, 1]
range-image domain, so a step of eps moves depth by eps/2 * 56 m.

Example:
    python baseline_fgsm.py --surrogate cenet --from_run cond_adv_cn_n1024 -n 1024 --frid_n 1024 --kitti_root $KITTI --kitti_seq 08 --also_range ours_s3=cond_adv_cn_n1024/adv_ranges.pt -o fgsm_cn_n1024
"""

import os
import sys
import json
import argparse
from glob import glob
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_guided as ca
from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.eval import build_model, DATASET_CONFIG, AGG_TYPE, MODAL2BATCHSIZE
from lidm.eval.metric_utils import preprocess_range, range2xyz as frid_range2xyz
from lidm.eval.eval_utils import compute_fd


def fgsm_step(surrogate, surrogate_type, clean, labels, eps,
              attack_mode, target_map, device):
    """One FGSM step on the range image; empty pixels (encoded as -1) are left unchanged."""
    clean_dev = clean.to(device)
    x = clean_dev.clone().detach().requires_grad_(True)
    valid = (clean_dev > -1.0 + 1e-3).float()
    if surrogate_type == 'rangenet':
        x_in, _ = ca.prepare_for_rangenet(x, device)
    else:
        x_in, _ = ca.prepare_for_cenet(x, device)
    scores, is_logprob = ca.net_forward(surrogate, x_in, surrogate_type)
    loss = ca.attack_loss(scores.float(), is_logprob, labels, attack_mode, target_map)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign() * valid).clamp(-1.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def range_to_frid_input(range_batch, cfg):
    """[B,1,H,W] in [-1,1] -> list of [4,H,W] (depth, x, y, z) in meters, built like lidm preprocess_range."""
    r01 = (range_batch.squeeze(1).float().cpu().numpy() + 1.0) / 2.0
    dmax = float(cfg['depth_range'][1])
    dmin = float(cfg['depth_range'][0])
    out = []
    for b in range(r01.shape[0]):
        depth = r01[b] * dmax
        depth[(depth <= dmin) | (depth >= dmax)] = -1.0        # mark empties as lidm pcd2range does
        xyz = frid_range2xyz(depth, log_scale=False, **cfg)
        out.append(np.vstack([depth[None], xyz]).astype(np.float32))
    return out


def pcds_to_frid_input(pcds, cfg):
    """Real point clouds -> [4,H,W] meter range inputs via lidm.preprocess_range."""
    return [preprocess_range(p, **cfg).astype(np.float32) for p in pcds]


def extract_feats(model, inputs_4ch, bs):
    """List of [4,H,W] -> [N, D] RangeNet features (the FRID feature space)."""
    feats = []
    for i in range(0, len(inputs_4ch), bs):
        batch = torch.from_numpy(np.stack(inputs_4ch[i:i + bs])).float().cuda()
        out = model(batch, return_final_logits=True, agg_type=AGG_TYPE)
        feats.append(out)
    return np.vstack(feats)


def load_kitti_pcds(kitti_root, n, seq='08'):
    """Load up to n KITTI scans (xyz in meters) from <root>/sequences/<seq>/velodyne or <root>/<seq>/velodyne."""
    candidates = [
        os.path.join(kitti_root, 'sequences', seq, 'velodyne'),
        os.path.join(kitti_root, seq, 'velodyne'),
    ]
    files = []
    for d in candidates:
        files = sorted(glob(os.path.join(d, '*.bin')))
        if files:
            print(f'[frid] velodyne dir: {d}')
            break
    if not files:
        raise FileNotFoundError(
            'No KITTI .bin found. Tried:\n  ' + '\n  '.join(candidates) +
            f'\nSet --kitti_root to the dir containing either <seq>/velodyne/*.bin '
            f'or sequences/<seq>/velodyne/*.bin.')
    files = files[:n]
    pcds = []
    for f in files:
        scan = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3]
        pcds.append(scan)
    print(f'[frid] loaded {len(pcds)} real KITTI scans from seq {seq}')
    return pcds


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--surrogate', choices=['rangenet', 'cenet'], default='rangenet',
                   help='network used to craft FGSM (white-box); the other is transfer')
    p.add_argument('--attack_mode', choices=['gt_aware_untargeted', 'mask_targeted'],
                   default='gt_aware_untargeted')
    p.add_argument('--source_class', type=int, default=9)
    p.add_argument('--target_class', type=int, default=1)
    p.add_argument('--eps', type=float, nargs='+',
                   default=[0.015686, 0.031373, 0.062745, 0.075, 0.086, 0.10, 0.12],
                   help='FGSM budgets in the [-1,1] range-image domain')
    p.add_argument('--from_run', type=str, default=None,
                   help='load clean_ranges.pt and labels.pt from this run dir instead of generating')
    p.add_argument('--target_mse', type=float, default=0.0056,
                   help='reference MSE for the printed eps_matched')
    p.add_argument('-n', '--n_samples', type=int, default=500)
    p.add_argument('-b', '--batch_size', type=int, default=4)
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('-s', '--seed', type=int, default=42)
    p.add_argument('-o', '--output_dir', type=str, default='fgsm_results')
    p.add_argument('--skip_frid', action='store_true')
    p.add_argument('--kitti_root', type=str, required=True)
    p.add_argument('--kitti_seq', type=str, default='08')
    p.add_argument('--frid_n', type=int, default=500,
                   help='number of real and generated scans used for FRID')
    p.add_argument('--also_range', type=str, nargs='*', default=[],
                   help='extra NAME=PATH range tensors (.pt) to include in FRID')
    return p


def main():
    t0 = datetime.now()
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[info] device={device} surrogate={args.surrogate} eps={args.eps}')

    nets = {'rangenet': ca.load_rangenet(device), 'cenet': ca.load_cenet(device)}
    surrogate = nets[args.surrogate]

    eps_list = list(args.eps)
    cm = {net: {'clean': np.zeros((20, 20), np.int64)} for net in nets}
    for net in nets:
        for e in eps_list:
            cm[net][f'adv_{e}'] = np.zeros((20, 20), np.int64)
    all_clean, all_labels = [], []
    all_adv = {e: [] for e in eps_list}
    mse  = {e: [0.0, 0] for e in eps_list}
    linf = {e: [0.0, 0] for e in eps_list}

    def fgsm_and_eval(clean_b, labels_b):
        """Evaluate one batch clean, run FGSM for every eps and update the accumulators."""
        N = clean_b.shape[0]
        clean_b = clean_b.to(device)
        labels_b = labels_b.to(device)
        eval_mask = labels_b > 0
        l_v = labels_b[eval_mask].cpu().numpy()
        for net_name, net in nets.items():
            p_clean = ca.predict_labels(net, clean_b, net_name, device)
            ca.accumulate_confusion(cm[net_name]['clean'], l_v,
                                    p_clean[eval_mask].cpu().numpy())
        target_map = ca.build_target_map(labels_b, args.attack_mode,
                                          args.source_class, args.target_class)
        for e in eps_list:
            adv = fgsm_step(surrogate, args.surrogate, clean_b, labels_b, e,
                            args.attack_mode, target_map, device)
            d = clean_b - adv
            mse[e][0]  += (d ** 2).mean().item() * N
            mse[e][1]  += N
            linf[e][0] += d.abs().flatten(1).amax(dim=1).mean().item() * N
            linf[e][1] += N
            for net_name, net in nets.items():
                p_adv = ca.predict_labels(net, adv, net_name, device)
                ca.accumulate_confusion(cm[net_name][f'adv_{e}'], l_v,
                                        p_adv[eval_mask].cpu().numpy())
            all_adv[e].append(adv.cpu())
        all_clean.append(clean_b.cpu())
        all_labels.append(labels_b.cpu())

    n_done = 0
    if args.from_run:
        clean_src = torch.load(os.path.join(args.from_run, 'clean_ranges.pt'),
                               weights_only=False)
        lab_src   = torch.load(os.path.join(args.from_run, 'labels.pt'),
                               weights_only=False)
        N_avail = min(len(clean_src), len(lab_src))
        if len(clean_src) != len(lab_src):
            raise ValueError(f'clean_ranges.pt ({len(clean_src)}) and labels.pt '
                             f'({len(lab_src)}) length mismatch in {args.from_run}')
        n_use = min(args.n_samples, N_avail)
        print(f'[from_run] loaded {n_use} matched (clean, labels) from {args.from_run}')
        for i in range(0, n_use, args.batch_size):
            cb = clean_src[i:i + args.batch_size]
            lb = lab_src  [i:i + args.batch_size]
            fgsm_and_eval(cb, lb)
            n_done += cb.shape[0]
            print(f'[fgsm] {n_done}/{n_use}')
    else:
        logdir = '/'.join(ca.SEM2LIDAR_CKPT.split('/')[:-1])
        config = OmegaConf.load(os.path.join(logdir, 'config.yaml'))
        print(f'[sem2lidar] loading {ca.SEM2LIDAR_CKPT} ...')
        pl_sd = torch.load(ca.SEM2LIDAR_CKPT, map_location='cpu', weights_only=False)
        model = instantiate_from_config(config.model)
        model.load_state_dict(pl_sd['state_dict'], strict=False)
        model = model.to(device).eval()
        loader, _ = ca.build_dataloader(config, args.batch_size)
        C_lat = model.channels
        H_lat, W_lat = model.image_size

        for batch in loader:
            if n_done >= args.n_samples:
                break
            z0, c, x, xrec, xc = model.get_input(
                batch, model.first_stage_key, return_first_stage_outputs=True,
                force_c_encode=True, return_original_cond=True, bs=args.batch_size)
            N = z0.shape[0]
            cond_raw = ca.find_conditioning(batch)
            if not torch.is_tensor(cond_raw):
                cond_raw = torch.stack(cond_raw, 0)
            labels_b = ca.conditioning_to_labels(cond_raw)[:N].to(device)

            shape = (N, C_lat, H_lat, W_lat)
            z_T = torch.randn(shape, device=device)
            clean_b = ca.cond_ddim_sample(model, surrogate, args.surrogate, c, labels_b, shape,
                                          ddim_steps=args.ddim_steps, eta=args.eta,
                                          adversarial=False, z_T=z_T)
            fgsm_and_eval(clean_b, labels_b)
            n_done += N
            print(f'[gen] {n_done}/{args.n_samples}')

    clean_t = torch.cat(all_clean, 0)[:args.n_samples]
    lab_t   = torch.cat(all_labels, 0)[:args.n_samples]
    torch.save(clean_t, os.path.join(args.output_dir, 'clean_ranges.pt'))
    torch.save(lab_t,   os.path.join(args.output_dir, 'labels.pt'))
    adv_t = {}
    for e in eps_list:
        adv_t[e] = torch.cat(all_adv[e], 0)[:args.n_samples]
        torch.save(adv_t[e], os.path.join(args.output_dir, f'adv_eps{e:.6f}.pt'))

    valid_frac = (clean_t > -1.0 + 1e-3).float().mean().item()
    eps_matched = (args.target_mse / max(valid_frac, 1e-6)) ** 0.5
    print(f'[valid] empirical valid_frac = {valid_frac:.3f}')
    print(f'[valid] for target MSE={args.target_mse}, eps_matched = '
          f'sqrt(MSE/valid_frac) = {eps_matched:.4f}  (~{eps_matched*255:.1f}/255)')

    report = ['FGSM (single-step) on conditional sem2lidar range images',
              f'surrogate={args.surrogate}  attack_mode={args.attack_mode}',
              f'eps (in [-1,1]): {eps_list}',
              f'n_samples={n_done}  seed={args.seed}', '']
    role = {args.surrogate: 'WHITE-BOX (FGSM)',
            ('cenet' if args.surrogate == 'rangenet' else 'rangenet'): 'TRANSFER'}
    miou_summary = {}
    for net_name in nets:
        ic, mc = ca.per_class_iou(cm[net_name]['clean'])
        miou_summary[net_name] = {'clean': mc}
        for e in eps_list:
            ia, ma = ca.per_class_iou(cm[net_name][f'adv_{e}'])
            miou_summary[net_name][e] = ma
            report.append(ca.format_table(f'{net_name} [{role[net_name]}] eps={e:.5f}',
                                          ic, mc, ia, ma))

    report.append('\n=== SUMMARY (mIoU and perturbation by eps) ===')
    report.append(f'[clean: valid_frac={valid_frac:.3f}  '
                  f'eps_matched(target MSE={args.target_mse})={eps_matched:.4f}'
                  f' (~{eps_matched*255:.1f}/255)]')
    report.append(f'{"eps":>10s} {"depth_m":>8s} {"MSE":>9s} {"Linf":>8s} '
                  f'{"RN_mIoU":>9s} {"CN_mIoU":>9s}')
    report.append(f'{"clean":>10s} {"-":>8s} {"-":>9s} {"-":>8s} '
                  f'{100*miou_summary["rangenet"]["clean"]:9.2f} '
                  f'{100*miou_summary["cenet"]["clean"]:9.2f}')
    for e in eps_list:
        m = mse[e][0] / max(mse[e][1], 1)
        li = linf[e][0] / max(linf[e][1], 1)
        report.append(f'{e:10.5f} {e/2*56:8.2f} {m:9.5f} {li:8.4f} '
                      f'{100*miou_summary["rangenet"][e]:9.2f} '
                      f'{100*miou_summary["cenet"][e]:9.2f}')

    frid_scores = {}
    if not args.skip_frid:
        try:
            cfg = dict(DATASET_CONFIG['kitti'])
            bs = MODAL2BATCHSIZE['range']
            feat_model = build_model('kitti', 'rangenet', device='cuda')

            ref_pcds = load_kitti_pcds(args.kitti_root, args.frid_n, args.kitti_seq)
            ref_feats = extract_feats(feat_model, pcds_to_frid_input(ref_pcds, cfg), bs)

            names, feats_list = [], []
            names.append('clean')
            feats_list.append(extract_feats(feat_model,
                              range_to_frid_input(clean_t[:args.frid_n], cfg), bs))
            for e in eps_list:
                names.append(f'adv_eps{e:.5f}')
                feats_list.append(extract_feats(feat_model,
                                  range_to_frid_input(adv_t[e][:args.frid_n], cfg), bs))
            for spec in args.also_range:
                nm, pth = spec.split('=', 1) if '=' in spec else spec.split(':', 1)
                t = torch.load(pth, weights_only=False)
                names.append(nm)
                feats_list.append(extract_feats(feat_model,
                                  range_to_frid_input(t[:args.frid_n], cfg), bs))

            print(f'[frid] real ref={len(ref_pcds)} | scoring {len(names)} sets ...')
            for nm, fe in zip(names, feats_list):
                frid_scores[nm] = float(compute_fd(ref_feats, fe))
            report.append('\n=== FRID vs real KITTI (lower = more realistic) ===')
            for nm in names:
                report.append(f'{nm:>16s}: {frid_scores[nm]:8.3f}')
        except Exception as ex:
            report.append(f'\n[FRID skipped: {ex}]')
            print(f'[frid] ERROR: {ex}')

    report.append(f'\nelapsed: {datetime.now() - t0}')
    text = '\n'.join(report)
    print(text)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(text)
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump({'eps': eps_list,
                   'valid_frac': valid_frac,
                   'eps_matched': eps_matched,
                   'target_mse': args.target_mse,
                   'miou': {k: {str(kk): vv for kk, vv in v.items()}
                            for k, v in miou_summary.items()},
                   'mse':  {str(e): mse[e][0]  / max(mse[e][1], 1)  for e in eps_list},
                   'linf': {str(e): linf[e][0] / max(linf[e][1], 1) for e in eps_list},
                   'frid': frid_scores}, f, indent=2)
    print(f'\n[done] saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
