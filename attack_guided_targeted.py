"""Targeted variant of attack_guided.py, used for Figure 3.

It replaces attack_loss so that in mask_targeted mode the loss covers only the
source-class pixels, then runs attack_guided.main() with the same arguments.

Example:
    python attack_guided_targeted.py --surrogate cenet --attack_mode mask_targeted --source_class 9 --target_class 11 --guidance_scale 6.0 --guidance_start 0.5 -n 10 -s 42 -o targeted_road_to_sidewalk
"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
import attack_guided as ca


def attack_loss_surgical(scores, is_logprob, labels, attack_mode, target_map):
    if attack_mode == 'gt_aware_untargeted':
        return ca.seg_loss(scores, is_logprob, labels)
    elif attack_mode == 'mask_targeted':
        source_mask = (labels != target_map)
        if source_mask.sum() == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        if is_logprob:
            pix_ce = F.nll_loss(scores, target_map, ignore_index=0, reduction='none')
        else:
            pix_ce = F.cross_entropy(scores, target_map, ignore_index=0, reduction='none')
        return -pix_ce[source_mask].mean()
    raise ValueError(f'unknown attack_mode {attack_mode}')


# cond_ddim_sample looks up ca.attack_loss at call time
ca.attack_loss = attack_loss_surgical

if __name__ == '__main__':
    ca.main()