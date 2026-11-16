"""SegPGD baseline (Gu et al., ECCV 2022) on sem2lidar range images.

For each budget eps, the script runs T SegPGD iterations on the clean range images
and evaluates CENet and RangeNet++ on the clean and perturbed images against the
conditioning labels. Empty pixels are not perturbed (disable with --no_valid_mask)
and class 0 (unlabeled) is ignored in the loss. FRID is not computed here, so run
eval_frid.py on the output folder.

Example:
    python baseline_segpgd.py --surrogate cenet --from_run cond_adv_cn_n1024 -n 1024 --T 20 --alpha 0.01 --eps 0.015686 0.031373 0.062745 0.075 0.086 0.12 0.165 0.20 -o segpgd_cn_n1024
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_guided as ca
from lidm.utils.misc_utils import instantiate_from_config, set_seed


def segpgd_step(surrogate, surrogate_type, clean, labels, eps, alpha, T,
                device, valid_mask_on=True, random_init=True):
    """Untargeted SegPGD on a batch: clean [B,1,H,W] in [-1,1], labels [B,H,W] with 0 = unlabeled."""
    clean_dev = clean.to(device)
    labels_dev = labels.to(device)
    valid = ((clean_dev > -1.0 + 1e-3).float()
             if valid_mask_on else torch.ones_like(clean_dev))

    # random init: x_0 = x_clean + U(-eps, eps)
    if random_init:
        noise = (torch.rand_like(clean_dev) * 2.0 - 1.0) * eps * valid
        x = clean_dev + noise
    else:
        x = clean_dev.clone()
    x = torch.max(torch.min(x, clean_dev + eps), clean_dev - eps).clamp(-1.0, 1.0)

    valid_label = labels_dev > 0

    for t in range(1, T + 1):
        x = x.detach().requires_grad_(True)

        if surrogate_type == 'rangenet':
            x_in, _ = ca.prepare_for_rangenet(x, device)
        else:
            x_in, _ = ca.prepare_for_cenet(x, device)
        scores, is_logprob = ca.net_forward(surrogate, x_in, surrogate_type)
        scores_f = scores.float()

        if is_logprob:
            ce = F.nll_loss(scores_f, labels_dev, ignore_index=0, reduction='none')
        else:
            ce = F.cross_entropy(scores_f, labels_dev, ignore_index=0, reduction='none')

        # split labeled pixels into correctly and wrongly classified
        with torch.no_grad():
            pred = scores_f.argmax(dim=1)
        correct = ((pred == labels_dev) & valid_label).float()
        wrong   = ((pred != labels_dev) & valid_label).float()

        # SegPGD loss weighting (Eq. 4, linear lambda from Eq. 5)
        lam = (t - 1) / (2.0 * T)
        denom = valid_label.sum().clamp_min(1).float()
        L = ((1.0 - lam) * (ce * correct).sum() + lam * (ce * wrong).sum()) / denom

        # sign step, then project to the eps-ball and to [-1, 1]
        grad = torch.autograd.grad(L, x)[0]
        x = (x + alpha * grad.sign() * valid).detach()
        x = torch.max(torch.min(x, clean_dev + eps), clean_dev - eps).clamp(-1.0, 1.0)
        # keep empty pixels exactly equal to clean
        if valid_mask_on:
            x = torch.where(valid > 0, x, clean_dev)

    return x.detach()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--surrogate', choices=['rangenet', 'cenet'], default='cenet',
                   help='network whose gradients guide the attack')
    p.add_argument('--eps', type=float, nargs='+',
                   default=[0.015686, 0.031373, 0.062745, 0.075, 0.086, 0.10, 0.12],
                   help='SegPGD budgets in the [-1,1] range-image domain')
    p.add_argument('--T', type=int, default=20,
                   help='number of attack iterations (T=1 gives SegFGSM)')
    p.add_argument('--alpha', type=float, default=0.01,
                   help='step size per iteration')
    p.add_argument('--no_random_init', action='store_true',
                   help='disable the U(-eps, eps) random init')
    p.add_argument('--no_valid_mask', action='store_true',
                   help='also perturb empty pixels')
    p.add_argument('--from_run', type=str, default=None,
                   help='load clean_ranges.pt and labels.pt from this run dir instead of generating')
    p.add_argument('--target_mse', type=float, default=0.0056,
                   help='reference MSE for the printed eps_matched')
    p.add_argument('-n', '--n_samples', type=int, default=500)
    p.add_argument('-b', '--batch_size', type=int, default=4)
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('-s', '--seed', type=int, default=42)
    p.add_argument('-o', '--output_dir', type=str, default='segpgd_results')
    return p


def main():
    t0 = datetime.now()
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[info] device={device} surrogate={args.surrogate} '
          f'T={args.T} alpha={args.alpha} eps={args.eps}')
    print(f'[info] valid_mask={"OFF" if args.no_valid_mask else "ON"}  '
          f'random_init={"OFF" if args.no_random_init else "ON"}')

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

    def segpgd_and_eval(clean_b, labels_b):
        """Evaluate one batch clean, run SegPGD for every eps and update the accumulators."""
        N = clean_b.shape[0]
        clean_b = clean_b.to(device)
        labels_b = labels_b.to(device)
        eval_mask = labels_b > 0
        l_v = labels_b[eval_mask].cpu().numpy()
        for net_name, net in nets.items():
            p_clean = ca.predict_labels(net, clean_b, net_name, device)
            ca.accumulate_confusion(cm[net_name]['clean'], l_v,
                                    p_clean[eval_mask].cpu().numpy())
        for e in eps_list:
            adv = segpgd_step(surrogate, args.surrogate, clean_b, labels_b,
                              e, args.alpha, args.T, device,
                              valid_mask_on=not args.no_valid_mask,
                              random_init=not args.no_random_init)
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
            del adv, d
        all_clean.append(clean_b.cpu())
        all_labels.append(labels_b.cpu())
        torch.cuda.empty_cache()

    n_done = 0
    if args.from_run:
        clean_src = torch.load(os.path.join(args.from_run, 'clean_ranges.pt'),
                               weights_only=False)
        lab_src   = torch.load(os.path.join(args.from_run, 'labels.pt'),
                               weights_only=False)
        if len(clean_src) != len(lab_src):
            raise ValueError(f'clean ({len(clean_src)}) / labels ({len(lab_src)}) '
                             f'length mismatch in {args.from_run}')
        n_use = min(args.n_samples, len(clean_src))
        print(f'[from_run] loaded {n_use} matched (clean, labels) from {args.from_run}')
        for i in range(0, n_use, args.batch_size):
            cb = clean_src[i:i + args.batch_size]
            lb = lab_src  [i:i + args.batch_size]
            segpgd_and_eval(cb, lb)
            n_done += cb.shape[0]
            print(f'[segpgd] {n_done}/{n_use}')
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
            clean_b = ca.cond_ddim_sample(model, surrogate, args.surrogate, c,
                                          labels_b, shape,
                                          ddim_steps=args.ddim_steps, eta=args.eta,
                                          adversarial=False, z_T=z_T)
            segpgd_and_eval(clean_b, labels_b)
            n_done += N
            print(f'[gen+segpgd] {n_done}/{args.n_samples}')

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

    report = ['SegPGD (Gu et al., ECCV 2022) on conditional sem2lidar range images',
              f'surrogate={args.surrogate}  T={args.T}  alpha={args.alpha}',
              f'eps (in [-1,1]): {eps_list}',
              f'valid_mask={"OFF" if args.no_valid_mask else "ON"}  '
              f'random_init={"OFF" if args.no_random_init else "ON"}',
              f'n_samples={n_done}  seed={args.seed}', '']
    role = {args.surrogate: 'WHITE-BOX (SegPGD)',
            ('cenet' if args.surrogate == 'rangenet' else 'rangenet'): 'TRANSFER'}
    miou_summary = {}
    for net_name in nets:
        ic, mc = ca.per_class_iou(cm[net_name]['clean'])
        miou_summary[net_name] = {'clean': mc}
        for e in eps_list:
            ia, ma = ca.per_class_iou(cm[net_name][f'adv_{e}'])
            miou_summary[net_name][e] = ma
            report.append(ca.format_table(
                f'{net_name} [{role[net_name]}] eps={e:.5f} T={args.T}',
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
    report.append(f'\nelapsed: {datetime.now() - t0}')
    report.append('\n[note] FRID skipped here -- run eval_frid.py on this dir to add it.')

    text = '\n'.join(report)
    print(text)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(text)
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump({'eps': eps_list, 'T': args.T, 'alpha': args.alpha,
                   'valid_frac': valid_frac, 'eps_matched': eps_matched,
                   'target_mse': args.target_mse,
                   'random_init': not args.no_random_init,
                   'valid_mask': not args.no_valid_mask,
                   'miou': {k: {str(kk): vv for kk, vv in v.items()}
                            for k, v in miou_summary.items()},
                   'mse':  {str(e): mse[e][0]  / max(mse[e][1], 1)  for e in eps_list},
                   'linf': {str(e): linf[e][0] / max(linf[e][1], 1) for e in eps_list}},
                  f, indent=2)
    print(f'\n[done] saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
