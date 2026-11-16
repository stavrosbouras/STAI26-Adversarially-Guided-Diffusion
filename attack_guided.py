"""Adversarially guided sampling with the LiDM sem2lidar model.

For each SemanticKITTI conditioning map, the script generates a clean and an adversarial
range image from the same initial noise, and in the later DDIM steps pushes the
adversarial one along the gradient of a segmentation loss from a surrogate network
(CENet or RangeNet++). Both networks are then evaluated on both images against the
conditioning labels, which gives the white-box and the transfer mIoU drop in one run.
The paired images, labels, metrics.txt and report.json are saved to the output folder.

Example:
    python attack_guided.py --surrogate cenet -n 1024 --guidance_scale 3.0 --guidance_start 0.5 -o cond_adv_cn_n1024
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime

sys.path.append('./')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.eval import build_model
from segnets.cenet import ResNet_34 as CENet, BasicConv2d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


SEM2LIDAR_CKPT = 'models/lidm/kitti/sem2lidar/model.ckpt'
RANGENET_HEAD  = 'pretrained_weights/kitti/rangenet/segmentation_head'
CENET_WEIGHTS  = 'pretrained_weights/kitti/cenet/SalsaNext_valid_best'   # 5-channel CENet

# sem2lidar uses linear depth: depth = range_01 * 56 m (log_scale=false)
FOV         = [3.0, -25.0]
DEPTH_SCALE = 56.0
DEPTH_MIN   = 1.0    # valid-pixel threshold (meters)

SEMANTICKITTI_CLASSES = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
    'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
    'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign',
]

# Per-network input normalization (channel order: range, x, y, z[, remission])
RANGENET_MEAN = np.array([12.12, 10.88, 0.23, -1.04], dtype=np.float32)
RANGENET_STD  = np.array([12.32, 11.47, 6.91,  0.86], dtype=np.float32)
CENET_MEAN    = np.array([11.71279, -0.1023471, 0.4952, -1.0545, 0.2877], dtype=np.float32)
CENET_STD     = np.array([10.24,    12.295865, 9.4287, 0.8643, 0.1450], dtype=np.float32)


def range2xyz(range_01, fov=FOV, depth_scale=DEPTH_SCALE):
    """Range [B,64,1024] in [0,1] -> [B,4,64,1024] (depth, x, y, z) with linear depth."""
    if range_01.dim() == 4:
        range_01 = range_01.squeeze(1)
    B, H, W = range_01.shape
    device = range_01.device
    fov_up   = fov[0] / 180.0 * math.pi
    fov_down = fov[1] / 180.0 * math.pi
    fov_range = abs(fov_down) + abs(fov_up)
    depth = range_01 * depth_scale
    sx = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, -1)
    sy = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, -1, W)
    yaw = math.pi * (sx * 2 - 1)
    pitch = (1.0 - sy) * fov_range - abs(fov_down)
    x = torch.cos(yaw) * torch.cos(pitch) * depth
    y = -torch.sin(yaw) * torch.cos(pitch) * depth
    z = torch.sin(pitch) * depth
    return torch.stack([depth, x, y, z], dim=1)


def prepare_for_rangenet(range_img_raw, device):
    """LiDM range [B,1,64,1024] in [-1,1] -> normalized RangeNet input [B,4,64,1024]."""
    range_01 = (range_img_raw.squeeze(1) + 1.0) / 2.0
    dxz = range2xyz(range_01).to(device)
    valid = (dxz[:, 0:1] > DEPTH_MIN).float()
    mean = torch.from_numpy(RANGENET_MEAN).to(device).view(1, 4, 1, 1)
    std  = torch.from_numpy(RANGENET_STD).to(device).view(1, 4, 1, 1)
    x = dxz * valid + mean * (1.0 - valid)
    return (x - mean) / std, valid


def prepare_for_cenet(range_img_raw, device):
    """LiDM range [B,1,64,1024] in [-1,1] -> normalized CENet input [B,5,64,1024], remission set to its mean."""
    range_01 = (range_img_raw.squeeze(1) + 1.0) / 2.0
    dxz = range2xyz(range_01)
    valid = (dxz[:, 0:1] > DEPTH_MIN).float()
    B, _, H, W = dxz.shape
    mean = torch.from_numpy(CENET_MEAN).to(device).view(1, 5, 1, 1)
    std  = torch.from_numpy(CENET_STD).to(device).view(1, 5, 1, 1)
    rem = torch.full((B, 1, H, W), float(CENET_MEAN[4]), device=dxz.device, dtype=dxz.dtype)
    x = torch.cat([dxz, rem], dim=1).to(device)
    valid = valid.to(device)
    x = x * valid + mean * (1.0 - valid)
    return (x - mean) / std, valid


class RangeNetWithHead(nn.Module):
    def __init__(self, base_model, head_weights_path):
        super().__init__()
        self.backbone = base_model.backbone
        self.decoder  = base_model.decoder
        self.head     = nn.Conv2d(32, 20, kernel_size=3, padding=1)
        head_dict = torch.load(head_weights_path, map_location='cpu')
        self.head.weight.data = head_dict['1.weight']
        self.head.bias.data   = head_dict['1.bias']

    def forward(self, x):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        return self.head(y)


def load_rangenet(device):
    base = build_model('kitti', 'rangenet', device='cpu')
    m = RangeNetWithHead(base, RANGENET_HEAD).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_cenet(device):
    m = CENet(nclasses=20, params={'train': {'aux_loss': {'use': True}}}).to(device)
    ckpt = torch.load(CENET_WEIGHTS, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def net_forward(model, x, net_type):
    """Return (scores, is_logprob): log-probs for CENet, which outputs softmax, and raw logits for RangeNet."""
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if net_type == 'cenet':
        return torch.log(out.clamp_min(1e-8)), True
    return out, False


def seg_loss(scores, is_logprob, target):
    """Cross-entropy on logits (RangeNet) or NLL on log-probs (CENet), ignoring class 0."""
    if is_logprob:
        return F.nll_loss(scores, target, ignore_index=0)
    return F.cross_entropy(scores, target, ignore_index=0)


def differentiable_decode(model, z):
    # decode without the VQ step so gradients reach z
    z_scaled = 1.0 / model.scale_factor * z
    quant = model.first_stage_model.post_quant_conv(z_scaled)
    dec = model.first_stage_model.decoder(quant)
    return dec


def build_target_map(labels, attack_mode, source_class, target_class):
    """mask_targeted: labels with source_class replaced by target_class; otherwise the labels."""
    if attack_mode == 'mask_targeted':
        T = labels.clone()
        T[labels == source_class] = target_class
        return T
    return labels


def attack_loss(scores, is_logprob, labels, attack_mode, target_map):
    """Loss to maximize: CE to the true labels (untargeted) or negative CE to the target map (mask_targeted)."""
    if attack_mode == 'gt_aware_untargeted':
        return seg_loss(scores, is_logprob, labels)
    elif attack_mode == 'mask_targeted':
        return -seg_loss(scores, is_logprob, target_map)
    raise ValueError(f'unknown attack_mode {attack_mode}')


def cond_ddim_sample(model, surrogate, surrogate_type, cond, labels, shape,
                     ddim_steps=50, eta=0.0,
                     adversarial=False, guidance_scale=3.0, guidance_start=0.5,
                     attack_mode='gt_aware_untargeted', target_map=None,
                     z_T=None, verbose=False):
    from lidm.models.diffusion.ddim import DDIMSampler
    device = model.device
    B, C, H, W = shape

    ddim = DDIMSampler(model)
    ddim.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta)
    timesteps = np.flip(ddim.ddim_timesteps)
    total_steps = len(timesteps)

    if z_T is None:
        z_T = torch.randn(shape, device=device)
    z = z_T.clone()

    iterator = enumerate(timesteps)
    if verbose:
        iterator = tqdm(iterator, total=total_steps, desc='cond-DDIM' + (' (adv)' if adversarial else ''))

    for step_idx, step in iterator:
        index = total_steps - step_idx - 1
        t = torch.full((B,), step, device=device, dtype=torch.long)

        with torch.no_grad():
            e_t = model.apply_model(z, t, cond)
            a_t       = torch.full((B, 1, 1, 1), ddim.ddim_alphas[index], device=device)
            a_prev    = torch.full((B, 1, 1, 1), ddim.ddim_alphas_prev[index], device=device)
            sigma_t   = torch.full((B, 1, 1, 1), ddim.ddim_sigmas[index], device=device)
            sqrt_1mat = torch.full((B, 1, 1, 1), ddim.ddim_sqrt_one_minus_alphas[index], device=device)
            pred_x0 = (z - sqrt_1mat * e_t) / a_t.sqrt()
            dir_xt  = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise   = sigma_t * torch.randn_like(z) if eta > 0 else 0.0
            z_prev  = a_prev.sqrt() * pred_x0 + dir_xt + noise

        progress = step_idx / total_steps
        if adversarial and progress >= guidance_start:
            torch.cuda.empty_cache()
            z_g = z_prev.detach().requires_grad_(True)
            range_img = differentiable_decode(model, z_g)
            if surrogate_type == 'rangenet':
                x_in, _ = prepare_for_rangenet(range_img, device)
            else:
                x_in, _ = prepare_for_cenet(range_img, device)
            scores, is_logprob = net_forward(surrogate, x_in, surrogate_type)
            loss = attack_loss(scores.float(), is_logprob, labels, attack_mode, target_map)
            grad = torch.autograd.grad(loss, z_g)[0]
            del range_img, x_in, scores, loss
            torch.cuda.empty_cache()
            gn = grad.norm()
            if gn > 0:
                grad = grad / gn
            z_prev = z_prev + guidance_scale * grad      # gradient ascent on attack_loss

        z = z_prev

    with torch.no_grad():
        final_range = model.decode_first_stage(z)
    return final_range


@torch.no_grad()
def predict_labels(model, range_img, net_type, device):
    if net_type == 'rangenet':
        x, _ = prepare_for_rangenet(range_img, device)
    else:
        x, _ = prepare_for_cenet(range_img, device)
    scores, _ = net_forward(model, x, net_type)
    return scores.argmax(dim=1)


def accumulate_confusion(cm, gt, pred, num_classes=20):
    flat = gt * num_classes + pred
    cm += np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class_iou(cm):
    ious = np.full(20, np.nan)
    for c in range(20):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        d = tp + fp + fn
        if d > 0:
            ious[c] = tp / d
    valid = ~np.isnan(ious[1:])
    return ious, (np.nanmean(ious[1:][valid]) if valid.any() else 0.0)


def format_table(name, ious_clean, miou_clean, ious_adv, miou_adv):
    lines = [f'\n=== {name} (clean -> adversarial) ===',
             f'{"Class":<16s} {"clean":>8s} {"adv":>8s} {"drop":>8s}']
    lines.append('-' * 44)
    for c in range(1, 20):
        cl = ious_clean[c]; ad = ious_adv[c]
        if np.isnan(cl) and np.isnan(ad):
            continue
        cl_s = f'{100*cl:7.2f}' if not np.isnan(cl) else '  N/A '
        ad_s = f'{100*ad:7.2f}' if not np.isnan(ad) else '  N/A '
        dr_s = f'{100*(cl-ad):+7.2f}' if not (np.isnan(cl) or np.isnan(ad)) else '   -  '
        lines.append(f'{SEMANTICKITTI_CLASSES[c]:<16s} {cl_s} {ad_s} {dr_s}')
    lines.append('-' * 44)
    lines.append(f'{"mIoU":<16s} {100*miou_clean:7.2f} {100*miou_adv:7.2f} {100*(miou_clean-miou_adv):+7.2f}')
    return '\n'.join(lines)


def build_dataloader(config, batch_size):
    data_config = config['data']['params']['validation']
    data_config['params'].update({
        'dataset_config': config['data']['params']['dataset'],
        'aug_config':     config['data']['params']['aug'],
        'return_pcd':     False,
        'max_objects_per_image': 5,
    })
    dataset = instantiate_from_config(data_config)

    def coll(data):
        out = {}
        for k in data[0].keys():
            v = [d[k] for d in data]
            if k not in ['reproj', 'raw']:
                try:
                    v = torch.from_numpy(np.stack(v, 0))
                except Exception:
                    v = [d[k] for d in data]
            out[k] = v
        return out

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        shuffle=False, drop_last=False, collate_fn=coll)
    return loader, dataset


def conditioning_to_labels(cond):
    if cond.dim() == 3:
        return cond.long()
    if cond.dim() == 4:
        return cond.argmax(dim=1).long() if cond.shape[1] > 1 else cond.squeeze(1).long()
    raise ValueError(f'unexpected conditioning shape {tuple(cond.shape)}')


def find_conditioning(batch):
    for k in ['segmentation', 'class', 'mask', 'label']:
        if k in batch and torch.is_tensor(batch[k]):
            return batch[k]
    raise KeyError(f'no conditioning key in batch: {list(batch.keys())}')


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--surrogate', choices=['rangenet', 'cenet'], default='rangenet',
                   help='Network whose gradients guide the attack.')
    p.add_argument('--attack_mode', choices=['gt_aware_untargeted', 'mask_targeted'],
                   default='gt_aware_untargeted')
    p.add_argument('--source_class', type=int, default=9, help='(mask_targeted) class to remap')
    p.add_argument('--target_class', type=int, default=1, help='(mask_targeted) class to remap to')
    p.add_argument('-n', '--n_samples', type=int, default=100)
    p.add_argument('-b', '--batch_size', type=int, default=4)
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('--guidance_scale', type=float, default=3.0)
    p.add_argument('--guidance_start', type=float, default=0.5)
    p.add_argument('-s', '--seed', type=int, default=42)
    p.add_argument('-o', '--output_dir', type=str, default='cond_adv_results')
    return p


def main():
    t_start = datetime.now()
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Device: {device} | surrogate={args.surrogate} | mode={args.attack_mode}')
    os.makedirs(args.output_dir, exist_ok=True)

    logdir = '/'.join(SEM2LIDAR_CKPT.split('/')[:-1])
    config = OmegaConf.load(os.path.join(logdir, 'config.yaml'))
    print(f'[sem2lidar] loading {SEM2LIDAR_CKPT} (~3.9 GB)...')
    pl_sd = torch.load(SEM2LIDAR_CKPT, map_location='cpu', weights_only=False)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd['state_dict'], strict=False)
    model = model.to(device).eval()

    loader, dataset = build_dataloader(config, args.batch_size)

    print('[load] RangeNet++  and  CENet (vanilla)')
    nets = {'rangenet': load_rangenet(device), 'cenet': load_cenet(device)}
    surrogate = nets[args.surrogate]

    # latent shape from model.channels, not diffusion_model.in_channels (that includes the conditioning)
    C_lat = model.channels
    H_lat, W_lat = model.image_size
    print(f'[info] latent shape per sample: [{C_lat}, {H_lat}, {W_lat}]')

    cm = {net: {'clean': np.zeros((20, 20), np.int64),
                'adv':   np.zeros((20, 20), np.int64)} for net in nets}

    all_clean, all_adv, all_labels = [], [], []
    mse_sum, mse_n = 0.0, 0
    n_done = 0

    log_cfg = {'sample': True, 'ddim_steps': args.ddim_steps, 'ddim_eta': args.eta,
               'quantize_denoised': False, 'inpaint': False,
               'plot_progressive_rows': False, 'plot_diffusion_rows': False, 'dset': dataset}

    for batch in loader:
        if n_done >= args.n_samples:
            break
        z0, c, x, xrec, xc = model.get_input(
            batch, model.first_stage_key, return_first_stage_outputs=True,
            force_c_encode=True, return_original_cond=True, bs=args.batch_size)
        N = z0.shape[0]
        cond_raw = find_conditioning(batch)
        if not torch.is_tensor(cond_raw):
            cond_raw = torch.stack(cond_raw, 0)
        labels = conditioning_to_labels(cond_raw)[:N].to(device)
        target_map = build_target_map(labels, args.attack_mode, args.source_class, args.target_class)

        shape = (N, C_lat, H_lat, W_lat)
        z_T = torch.randn(shape, device=device)    # same z_T for clean and adversarial

        clean = cond_ddim_sample(model, surrogate, args.surrogate, c, labels, shape,
                                 ddim_steps=args.ddim_steps, eta=args.eta,
                                 adversarial=False, z_T=z_T)
        adv = cond_ddim_sample(model, surrogate, args.surrogate, c, labels, shape,
                               ddim_steps=args.ddim_steps, eta=args.eta,
                               adversarial=True, guidance_scale=args.guidance_scale,
                               guidance_start=args.guidance_start,
                               attack_mode=args.attack_mode, target_map=target_map, z_T=z_T)

        eval_mask = labels > 0
        l_v = labels[eval_mask].cpu().numpy()
        for net_name, net in nets.items():
            p_clean = predict_labels(net, clean, net_name, device)
            p_adv   = predict_labels(net, adv,   net_name, device)
            accumulate_confusion(cm[net_name]['clean'], l_v, p_clean[eval_mask].cpu().numpy())
            accumulate_confusion(cm[net_name]['adv'],   l_v, p_adv[eval_mask].cpu().numpy())

        mse_sum += ((clean - adv) ** 2).mean().item() * N
        mse_n += N
        all_clean.append(clean.cpu()); all_adv.append(adv.cpu()); all_labels.append(labels.cpu())
        n_done += N
        print(f'[gen] {n_done}/{args.n_samples} samples done')

    clean_t = torch.cat(all_clean, 0)[:args.n_samples]
    adv_t   = torch.cat(all_adv, 0)[:args.n_samples]
    lab_t   = torch.cat(all_labels, 0)[:args.n_samples]
    torch.save(clean_t, os.path.join(args.output_dir, 'clean_ranges.pt'))
    torch.save(adv_t,   os.path.join(args.output_dir, 'adv_ranges.pt'))
    torch.save(lab_t,   os.path.join(args.output_dir, 'labels.pt'))

    report = [f'Conditional adversarial attack',
              f'surrogate={args.surrogate}  attack_mode={args.attack_mode}',
              f'guidance_scale={args.guidance_scale}  guidance_start={args.guidance_start}',
              f'n_samples={n_done}  ddim={args.ddim_steps}  eta={args.eta}  seed={args.seed}',
              f'clean-vs-adv range MSE: {mse_sum/max(mse_n,1):.6f}']
    if args.attack_mode == 'mask_targeted':
        report.append(f'target: {SEMANTICKITTI_CLASSES[args.source_class]} -> '
                      f'{SEMANTICKITTI_CLASSES[args.target_class]}')

    for net_name in nets:
        ic, mc = per_class_iou(cm[net_name]['clean'])
        ia, ma = per_class_iou(cm[net_name]['adv'])
        role = 'WHITE-BOX (guided)' if net_name == args.surrogate else 'TRANSFER'
        block = format_table(f'{net_name}  [{role}]', ic, mc, ia, ma)
        report.append(block)

    text = '\n'.join(report)
    print(text)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(text + '\n')

    meta = {'surrogate': args.surrogate, 'attack_mode': args.attack_mode,
            'source_class': args.source_class, 'target_class': args.target_class,
            'n_samples': n_done, 'ddim_steps': args.ddim_steps, 'eta': args.eta,
            'guidance_scale': args.guidance_scale, 'guidance_start': args.guidance_start,
            'seed': args.seed, 'clean_adv_mse': mse_sum / max(mse_n, 1),
            'timestamp_start': t_start.isoformat(), 'timestamp_end': datetime.now().isoformat()}
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n[done] saved clean/adv/labels + metrics.txt + report.json to {args.output_dir}/')


if __name__ == '__main__':
    main()
