"""Bootstrap confidence interval for the FRID gap between clean and adversarial samples.

Real, clean and adversarial features are resampled with replacement --bootstrap times,
and the script reports the 95% interval of FRID(real, clean) - FRID(real, adversarial).
Output goes to frid_bootstrap_results.txt next to --yours, or to -o.

Example:
    python eval_frid_bootstrap.py --clean cond_adv_cn_n1024/clean_ranges.pt --yours cond_adv_cn_n1024/adv_ranges.pt --kitti_root $KITTI --kitti_seq 08 --n 1024 --bootstrap 500
"""
import os
import sys
import argparse
import numpy as np
import torch

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
    p.add_argument('--clean', required=True, help='clean_ranges.pt')
    p.add_argument('--yours', required=True, help='adv_ranges.pt')
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--kitti_seq', default='08')
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--output', default=None,
                   help='text file for the printed results '
                        '(default: frid_bootstrap_results.txt next to --yours)')
    args = p.parse_args()

    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    np.random.seed(args.seed)
    cfg = dict(DATASET_CONFIG['kitti'])

    log('[extract] feature extractor...')
    feat_model = build_model('kitti', 'rangenet', device='cuda')
    feat_model.eval()

    log(f'[extract] real KITTI seq {args.kitti_seq} (n={args.n})...')
    ref_pcds = load_kitti_pcds(args.kitti_root, args.n, args.kitti_seq)
    ref_feats = extract_feats_safe(
        feat_model,
        pcds_to_frid_input(ref_pcds, cfg),
        args.batch_size
    )

    log('[extract] clean...')
    clean_t = torch.load(args.clean, weights_only=False)[:args.n]
    clean_feats = extract_feats_safe(
        feat_model,
        range_to_frid_input(clean_t, cfg),
        args.batch_size
    )

    log('[extract] yours...')
    yours_t = torch.load(args.yours, weights_only=False)[:args.n]
    yours_feats = extract_feats_safe(
        feat_model,
        range_to_frid_input(yours_t, cfg),
        args.batch_size
    )

    del clean_t, yours_t, feat_model
    torch.cuda.empty_cache()

    frid_clean = float(compute_fd(ref_feats, clean_feats))
    frid_yours = float(compute_fd(ref_feats, yours_feats))

    log(f'\n[point] FRID(R, clean) = {frid_clean:.3f}')
    log(f'[point] FRID(R, yours) = {frid_yours:.3f}')
    log(f'[point] gap clean-yours = {frid_clean - frid_yours:+.3f}  (>0 => yours closer to real)\n')

    N = args.n
    fc = np.zeros(args.bootstrap)
    fy = np.zeros(args.bootstrap)
    gaps = np.zeros(args.bootstrap)

    log(f'[bootstrap] {args.bootstrap} resamples (n={N})...')

    for b in range(args.bootstrap):
        ir = np.random.choice(N, N, replace=True)
        ic = np.random.choice(N, N, replace=True)
        iy = np.random.choice(N, N, replace=True)

        fc[b] = compute_fd(ref_feats[ir], clean_feats[ic])
        fy[b] = compute_fd(ref_feats[ir], yours_feats[iy])
        gaps[b] = fc[b] - fy[b]

        if (b + 1) % 100 == 0:
            log(f'  {b + 1}/{args.bootstrap}')

    q025, q975 = np.quantile(gaps, [0.025, 0.975])
    p_null = (gaps <= 0).mean()

    log('\n=== BOOTSTRAP RESULTS ===')
    log(f'FRID(R, clean):  {fc.mean():7.2f} +/- {fc.std():5.2f}  (point={frid_clean:.2f})')
    log(f'FRID(R, yours):  {fy.mean():7.2f} +/- {fy.std():5.2f}  (point={frid_yours:.2f})')
    log(f'Gap (clean-yours): {gaps.mean():+7.2f} +/- {gaps.std():5.2f}')
    log(f'95% CI for gap: [{q025:+.2f}, {q975:+.2f}]')
    log(f'P(gap <= 0 | bootstrap) = {p_null:.3f}')

    if q025 > 0:
        log('\nVERDICT: gap is STATISTICALLY SIGNIFICANT -- yours is genuinely closer to real.')
    elif q975 < 0:
        log('\nVERDICT: gap is STATISTICALLY SIGNIFICANT -- clean is genuinely closer to real.')
    else:
        log('\nVERDICT: gap is INDISTINGUISHABLE FROM NOISE (95% CI includes 0).')
        log('         The "yours < clean" finding is measurement noise.')

    out_path = args.output or os.path.join(os.path.dirname(args.yours) or '.',
                                           'frid_bootstrap_results.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[done] saved {out_path}')


if __name__ == '__main__':
    main()