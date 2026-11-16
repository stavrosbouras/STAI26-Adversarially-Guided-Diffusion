"""Paired Chamfer distance between clean and perturbed scans.

Range images are converted back to point clouds, and every set in the run directory
(adv_eps*.pt) and every --also_range tensor is compared with clean_ranges.pt scan by
scan using the squared Chamfer distance. The script also reports the per-scan JSD
between each scan's bird's-eye-view occupancy histogram and that of real SemanticKITTI
scans. Results go to geometry_metrics_v2.txt in the run directory, or to -o.

Example:
    python eval_chamfer.py --run_dir segpgd_cn_eps004_n1024 --kitti_root $KITTI --kitti_seq 08 --n 1024 --voxel 0.5 --cd_subsample 8192 -o segpgd_cn_eps004_n1024/geometry_metrics.txt
"""
import os, sys, argparse, numpy as np, torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, '.')
from baseline_fgsm import range_to_frid_input, load_kitti_pcds
from lidm.eval import DATASET_CONFIG


def range_to_pcd_list(range_batch, cfg):
    """[B,1,H,W] in [-1,1] -> list of [Npts,3] meter point clouds."""
    inputs = range_to_frid_input(range_batch, cfg)
    pcds = []
    for arr in inputs:
        depth = arr[0]; xyz = arr[1:4]
        valid = depth > 0
        pcds.append(xyz[:, valid].T.astype(np.float32))
    return pcds


def subsample(pcd, n, seed):
    if len(pcd) <= n: return pcd
    rng = np.random.default_rng(seed)
    return pcd[rng.choice(len(pcd), n, replace=False)]


def chamfer(x, y):
    """Symmetric squared Chamfer distance (LiDM/LiDARGen convention)."""
    tx = cKDTree(x); ty = cKDTree(y)
    d_yx = tx.query(y, k=1)[0]
    d_xy = ty.query(x, k=1)[0]
    return float((d_yx ** 2).mean() + (d_xy ** 2).mean())


def compute_cd_paired(a_list, b_list, n_sub, seed=0):
    assert len(a_list) == len(b_list)
    cds = []
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if len(a) == 0 or len(b) == 0: continue
        cds.append(chamfer(subsample(a, n_sub, seed + i),
                           subsample(b, n_sub, seed + i)))
    return float(np.mean(cds)) if cds else float('nan')


def bev_histogram_single(pcd, voxel=0.5, xr=(-50, 50), yr=(-50, 50)):
    """BEV occupancy histogram of one scan, normalized to a distribution."""
    nx = int((xr[1] - xr[0]) / voxel)
    ny = int((yr[1] - yr[0]) / voxel)
    g = np.zeros((nx, ny), dtype=np.float64)
    if len(pcd) == 0:
        return None
    m = ((pcd[:, 0] >= xr[0]) & (pcd[:, 0] < xr[1]) &
         (pcd[:, 1] >= yr[0]) & (pcd[:, 1] < yr[1]))
    ps = pcd[m]
    if len(ps) == 0:
        return None
    ix = np.clip(((ps[:, 0] - xr[0]) / voxel).astype(int), 0, nx - 1)
    iy = np.clip(((ps[:, 1] - yr[0]) / voxel).astype(int), 0, ny - 1)
    np.add.at(g, (ix, iy), 1)
    g = g.flatten()
    s = g.sum()
    return (g / s) if s > 0 else None


def bev_histogram_aggregate(pcds, voxel=0.5, xr=(-50, 50), yr=(-50, 50)):
    """Aggregate BEV histogram across many scans."""
    nx = int((xr[1] - xr[0]) / voxel)
    ny = int((yr[1] - yr[0]) / voxel)
    g = np.zeros((nx, ny), dtype=np.float64)
    for p in pcds:
        if len(p) == 0: continue
        m = ((p[:, 0] >= xr[0]) & (p[:, 0] < xr[1]) &
             (p[:, 1] >= yr[0]) & (p[:, 1] < yr[1]))
        ps = p[m]
        if len(ps) == 0: continue
        ix = np.clip(((ps[:, 0] - xr[0]) / voxel).astype(int), 0, nx - 1)
        iy = np.clip(((ps[:, 1] - yr[0]) / voxel).astype(int), 0, ny - 1)
        np.add.at(g, (ix, iy), 1)
    g = g.flatten()
    s = g.sum()
    return (g / s) if s > 0 else None


def compute_per_scan_jsd(ref_aggregate, samp_pcds, voxel=0.5, eps=1e-12):
    """Mean and std of the JSD between each scan's BEV histogram and the reference aggregate."""
    jsds = []
    r_smooth = (ref_aggregate + eps) / (1 + eps * len(ref_aggregate))
    for p in samp_pcds:
        h = bev_histogram_single(p, voxel=voxel)
        if h is None:
            continue
        h_smooth = (h + eps) / (1 + eps * len(h))
        jsds.append(jensenshannon(h_smooth, r_smooth))
    if not jsds:
        return float('nan'), float('nan')
    return float(np.mean(jsds)), float(np.std(jsds))


def main():
    P = argparse.ArgumentParser()
    P.add_argument('--run_dir', required=True)
    P.add_argument('--kitti_root', required=True)
    P.add_argument('--kitti_seq', default='08')
    P.add_argument('--n', type=int, default=1024)
    P.add_argument('--voxel', type=float, default=0.5)
    P.add_argument('--cd_subsample', type=int, default=8192)
    P.add_argument('--also_range', nargs='*', default=[])
    P.add_argument('-o', '--output', default=None)
    args = P.parse_args()

    cfg = dict(DATASET_CONFIG['kitti'])

    print(f'[load] real KITTI seq {args.kitti_seq} (n={args.n})...')
    ref_pcds = load_kitti_pcds(args.kitti_root, args.n, args.kitti_seq)

    print(f'[compute] real KITTI aggregate BEV histogram...')
    ref_aggregate = bev_histogram_aggregate(ref_pcds, voxel=args.voxel)

    sample_sets = []
    clean_pt = os.path.join(args.run_dir, 'clean_ranges.pt')
    clean_pcds = None
    if os.path.exists(clean_pt):
        print(f'[load] clean...')
        t = torch.load(clean_pt, weights_only=False)[:args.n]
        clean_pcds = range_to_pcd_list(t, cfg); del t
        sample_sets.append(('clean', clean_pcds))

    for f in sorted(os.listdir(args.run_dir)):
        if f.startswith('adv_eps') and f.endswith('.pt'):
            name = f.replace('.pt', '')
            print(f'[load] {name}...')
            t = torch.load(os.path.join(args.run_dir, f), weights_only=False)[:args.n]
            sample_sets.append((name, range_to_pcd_list(t, cfg))); del t

    for spec in args.also_range:
        nm, pth = spec.split('=', 1) if '=' in spec else spec.split(':', 1)
        print(f'[load] {nm} from {pth}...')
        t = torch.load(pth, weights_only=False)[:args.n]
        sample_sets.append((nm, range_to_pcd_list(t, cfg))); del t

    print('\n[compute] per-scan JSD vs aggregate real KITTI...')
    per_scan_jsd = {}
    for name, pcds in sample_sets:
        mean_jsd, std_jsd = compute_per_scan_jsd(ref_aggregate, pcds, voxel=args.voxel)
        per_scan_jsd[name] = (mean_jsd, std_jsd)
        print(f'  {name:>30s}: mean={mean_jsd:.5f}  std={std_jsd:.5f}')

    cd = {}
    if clean_pcds is not None:
        print(f'\n[compute] paired squared CD vs clean (subsample={args.cd_subsample})...')
        for name, pcds in sample_sets:
            if name == 'clean': continue
            cd[name] = compute_cd_paired(clean_pcds, pcds, args.cd_subsample)
            print(f'  {name:>30s}: CD = {cd[name]:.4f}')

    out_path = args.output or os.path.join(args.run_dir, 'geometry_metrics_v2.txt')
    lines = ['=== Per-scan geometric realism metrics (non-network) ===',
             f'reference: real KITTI seq {args.kitti_seq} (n={args.n})',
             f'BEV voxelization: {args.voxel} m, x/y in [-50, 50]',
             f'CD: paired squared Chamfer vs clean, {args.cd_subsample} points/scan',
             '',
             '--- Per-scan JSD vs aggregate real KITTI (realism, lower = better) ---',
             f'{"method":>30s} {"mean":>10s} {"std":>10s}']
    for k, (m, s) in per_scan_jsd.items():
        lines.append(f'{k:>30s} {m:10.5f} {s:10.5f}')
    if cd:
        lines += ['',
                  '--- Paired CD vs clean (perturbation magnitude in m^2) ---']
        for k, v in cd.items():
            lines.append(f'{k:>30s}: {v:8.4f}')
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(out_path, 'w') as f: f.write(txt)
    print(f'\n[done] saved {out_path}')


if __name__ == '__main__':
    main()