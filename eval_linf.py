"""Perturbation size between clean_ranges.pt and adv_ranges.pt in a run directory.

Reports the MSE, the per-sample L-infinity norm (mean, median and maximum) and the
fraction of pixels that change by more than fixed thresholds. Output goes to
linf_results.txt in the run directory, or to -o.

Example:
    python eval_linf.py --run_dir cond_adv_cn_n1024
"""
import os
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument('--run_dir', required=True,
               help='run directory containing clean_ranges.pt and adv_ranges.pt')
p.add_argument('-o', '--output', default=None,
               help='text file for the printed results (default: <run_dir>/linf_results.txt)')
args = p.parse_args()

clean = torch.load(os.path.join(args.run_dir, 'clean_ranges.pt'), weights_only=False)
adv = torch.load(os.path.join(args.run_dir, 'adv_ranges.pt'), weights_only=False)
d = (clean - adv).abs()

lines = []
def out(s=''):
    print(s)
    lines.append(s)

out(f'{args.run_dir} (n={len(clean)}):')
out(f'  MSE                                 = {(d**2).mean().item():.6f}')
out(f'  L_inf per sample, then mean         = {d.flatten(1).amax(dim=1).mean().item():.4f}')
out(f'  L_inf per sample, then median       = {d.flatten(1).amax(dim=1).median().item():.4f}')
out(f'  L_inf global (max across all)       = {d.max().item():.4f}')
out()
out(f'  fraction of pixels with |delta| > 0.05: {(d > 0.05).float().mean().item():.4%}')
out(f'  fraction of pixels with |delta| > 0.10: {(d > 0.10).float().mean().item():.4%}')
out(f'  fraction of pixels with |delta| > 0.20: {(d > 0.20).float().mean().item():.4%}')
out(f'  fraction of pixels with |delta| > 0.50: {(d > 0.50).float().mean().item():.4%}')

out_path = args.output or os.path.join(args.run_dir, 'linf_results.txt')
with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\n[done] saved {out_path}')
