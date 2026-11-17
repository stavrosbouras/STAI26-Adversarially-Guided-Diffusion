# Adversarially Guided Diffusion for LiDAR Range Image Synthesis

## 📝 Overview

This repository contains the official implementation of our paper:

**Authors:** Stavros Bouras, Antonios Makris, Alexandros Gkillas, Aris S. Lalos, Konstantinos Tserpes<br>
**Workshop:** 1st International Workshop on Secure and Trustworthy AI (STAI) @ ECML PKDD (2026)<br>
**Paper:** [Adversarially Guided Diffusion for LiDAR Range Image Synthesis](https://arxiv.org/abs/2607.09787)

## 🔍 Abstract

LiDAR semantic segmentation is a key perception task in autonomous driving, where false predictions can affect downstream planning and safety-critical decision-making. Although adversarial attacks, and specifically adversarial examples, have been widely studied for image classification and 3D point cloud segmentation, unrestricted adversarial examples remain largely unexplored in the space of 2D range images, which are projections of 3D point clouds. The proposed method is, to the best of our knowledge, the first diffusion-based unrestricted adversarial attack against 2D range-image segmentation, using adversarial guidance from a segmentation loss. By applying guidance directly during sampling, the method produces unrestricted adversarial examples that remain close to the learned LiDAR data manifold while inducing structured segmentation errors. Experiments on the SemanticKITTI dataset using RangeNet++ and CENet segmentation networks demonstrate that the attack provides adjustable degradation across guidance strengths and transfers across segmentation architectures. Compared with norm-bounded FGSM and SegPGD baselines, the proposed attack offers a distinct effectiveness–realism trade-off, achieving controllable white-box and transfer degradation while maintaining competitive distributional and visual realism.

## 🗂️ Project Structure

```text
.
├── attack_guided.py            # attack: adversarially guided sem2lidar sampling (clean + adversarial pair per conditioning map); mIoU for CENet and RangeNet++
├── attack_guided_targeted.py   # attack: targeted variant (source-class pixels only), used for Figure 3
├── baseline_fgsm.py            # baseline: FGSM on the clean range images (+ FRID)
├── baseline_segpgd.py          # baseline: SegPGD on the clean range images
├── eval_frid.py                # evaluation: FRID (RangeNet++ features) vs real SemanticKITTI
├── eval_frid_cenet.py          # evaluation: FRID with CENet features
├── eval_frid_bootstrap.py      # evaluation: bootstrap CI of the clean − adversarial FRID gap
├── eval_chamfer.py             # evaluation: paired Chamfer distance vs clean (and per-scan BEV JSD)
├── eval_linf.py                # evaluation: L∞ / MSE between clean and adversarial range images
├── eval_targeted.py            # evaluation: source-pixel reassignment percentages for the targeted run
├── segnets/
│   └── cenet.py                # CENet (ResNet-34) segmentation network, adapted from the original CENet code
├── patches/
│   └── lidm.patch              # patch applied to LiDM (torch.load(..., weights_only=False))
├── third_party/
│   └── LiDAR-Diffusion/        # LiDM (git submodule)
├── data -> third_party/LiDAR-Diffusion/data        # symlink to LiDM's data folder
├── configs -> third_party/LiDAR-Diffusion/configs  # symlink to LiDM's configs folder
├── weights.md5                 # MD5 checksums of the pretrained weights
├── requirements.txt            # Python dependencies
└── requirements-lock.txt       # exact package versions we tested
```

`models/`, `pretrained_weights/`, `dataset/` and the run output folders (`cond_adv_*/`, `fgsm_*/`, `segpgd_*/`, `targeted_*/`) are created by the user and are not tracked.

## 🚀 Quick Start

**Prerequisites:** Linux, conda, an NVIDIA GPU (we used one A100 80 GB; a 1024-sample generation run takes roughly 50 minutes).

### 1. Install Dependencies

Clone this repository with `--recursive`, or run `git submodule update --init` after cloning.

Tested with Python 3.10 and PyTorch built for CUDA 12.1.

```bash
conda create -n lidar_diffusion python=3.10 -y
conda activate lidar_diffusion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# taming-transformers (LiDM imports the `taming` package)
git clone https://github.com/CompVis/taming-transformers.git /tmp/taming-transformers
cp -r /tmp/taming-transformers/taming "$(python -c 'import site; print(site.getsitepackages()[0])')/"

# CLIP
pip install git+https://github.com/openai/CLIP.git

# patch LiDM and make it importable
git -C third_party/LiDAR-Diffusion apply ../../patches/lidm.patch
export PYTHONPATH=$PWD/third_party/LiDAR-Diffusion:$PYTHONPATH
```

On Google Colab, install conda first (`pip install -q condacolab`, then `import condacolab; condacolab.install()`) and run the commands through the environment, e.g. `conda run -n lidar_diffusion pip install ...`.

For the exact package versions we tested, replace the two `pip install` lines above with `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121`. taming-transformers is still copied manually as shown.

The repo root contains the relative symlinks `data -> third_party/LiDAR-Diffusion/data` and `configs -> third_party/LiDAR-Diffusion/configs`, so LiDM's data and config paths resolve when running from the repo root.

All commands below are run from the repository root.

### 2. Download Weights

| path (relative to the repo root) | model | source |
|---|---|---|
| `models/lidm/kitti/sem2lidar/model.ckpt` (+ its `config.yaml`) | LiDM, semantic map → LiDAR | [LiDM repo](https://github.com/hancyran/LiDAR-Diffusion), "Semantic Map to LiDAR" checkpoint |
| `models/first_stage_models/kitti/f_c2_p4_wo_logscale/model.ckpt` | LiDM first-stage autoencoder (loaded by the sem2lidar config) | [LiDM repo](https://github.com/hancyran/LiDAR-Diffusion), first-stage model zoo |
| `pretrained_weights/kitti/rangenet/` | RangeNet++ (transfer target, FRID feature extractor) | [LiDM repo](https://github.com/hancyran/LiDAR-Diffusion), evaluation toolbox model zoo (`lidm/eval/README.md`) |
| `pretrained_weights/kitti/cenet/SalsaNext_valid_best` | CENet (surrogate / white-box target) | [CENet repo](https://github.com/huixiancheng/CENet), SemanticKITTI checkpoint |

These are the exact files used in the paper. After downloading, check them from the repo root with `md5sum -c weights.md5`. Every line should say OK.

### 3. Prepare Data

We use **SemanticKITTI sequence 08** (validation split). Download it from [semantic-kitti.org](http://www.semantic-kitti.org/dataset.html). Evaluation scripts take the dataset path with `--kitti_root` (the directory that contains `sequences/08/velodyne/*.bin`):

```bash
export KITTI=/path/to/SemanticKITTI/dataset
```

The generators (`attack_guided.py`, and `baseline_fgsm.py` / `baseline_segpgd.py` without `--from_run`) load the semantic conditioning maps through LiDM's own data loader, which reads `./dataset/SemanticKITTI/dataset/sequences/...`. Place or symlink the dataset there, e.g. `mkdir -p dataset && ln -s /path/to/SemanticKITTI dataset/SemanticKITTI`, which also gives `KITTI=$PWD/dataset/SemanticKITTI/dataset`.

### 4. Run the Experiments

Results may differ slightly from the paper due to non-deterministic GPU operations. All runs use seed 42, deterministic DDIM (η = 0) and 50 steps (the script defaults).

#### Guided attack

```bash
python attack_guided.py --surrogate <cenet|rangenet> --guidance_scale <s> --guidance_start <τ*> -n 1024 -o <run_dir>
```

Paper: surrogate `cenet` (Tables 1–3) and `rangenet` (§4.3); s ∈ {3, 6, 10}; τ* = 0.5, plus τ* = 0.25 (s = 6) and τ* = 0.75 (s = 10) in §4.3.

#### Targeted attack

```bash
python attack_guided_targeted.py --surrogate cenet --attack_mode mask_targeted --source_class <id> --target_class <id> --guidance_scale 6.0 --guidance_start 0.5 -n 10 -o <run_dir>
```

Paper: road (9) → sidewalk (11).

#### Baselines

The baselines reuse the clean samples of a guided-attack run, so all methods see the same scenes. FRID is computed separately with `eval_frid.py`.

```bash
python baseline_fgsm.py --surrogate cenet --from_run <run_dir> -n 1024 --eps <ε ...> --kitti_root $KITTI --skip_frid -o <out_dir>
python baseline_segpgd.py --surrogate cenet --from_run <run_dir> -n 1024 --T 20 --alpha 0.01 --eps <ε ...> -o <out_dir>
```

Paper: FGSM ε ∈ {0.015686, 0.062745, 0.12}; SegPGD ε ∈ {0.004, 0.015686, 0.12}.

#### Evaluation

Each script reads a run folder. `eval_frid.py`, `eval_frid_cenet.py` and `eval_chamfer.py` automatically use `clean_ranges.pt` and the baseline samples (`adv_eps*.pt`) in it; for a guided-attack run, add its adversarial samples with `--also_range ours=<run_dir>/adv_ranges.pt`.

```bash
python eval_frid.py --run_dir <dir> --kitti_root $KITTI --kitti_seq 08 --frid_n 1024 [--also_range NAME=path ...]
python eval_frid_cenet.py   # same arguments; CENet features, for RangeNet++-guided runs
python eval_frid_bootstrap.py --clean <dir>/clean_ranges.pt --yours <dir>/adv_ranges.pt --kitti_root $KITTI --kitti_seq 08 --n 1024 --bootstrap 500
python eval_chamfer.py --run_dir <dir> --kitti_root $KITTI --kitti_seq 08 --n 1024 --voxel 0.5 --cd_subsample 8192 [--also_range NAME=path ...]
python eval_linf.py --run_dir <dir>
python eval_targeted.py --run_dir <dir> --source 9 --target 11 --seg_net cenet
```

| Reported metric | Source |
|---|---|
| WB / transfer drop, MSE | `metrics.txt` of the attack or baseline run (drop % = (clean − adv) / clean mIoU) |
| L∞ | `eval_linf.py` (ours) or `metrics.txt` (baselines) |
| FRID | `eval_frid.py` / `eval_frid_cenet.py` |
| FRID confidence interval | `eval_frid_bootstrap.py` |
| Chamfer distance | `eval_chamfer.py` |
| Targeted percentages | `eval_targeted.py` |

## 📖 Citation

If you use this code, please cite our paper:

```bibtex
@article{bouras2026adversarially,
  title={Adversarially Guided Diffusion for LiDAR Range Image Synthesis},
  author={Bouras, Stavros and Makris, Antonios and Gkillas, Alexandros and Lalos, Aris S. and Tserpes, Konstantinos},
  journal={arXiv preprint arXiv:2607.09787},
  year={2026}
}
```

## 🙏 Acknowledgment

This code builds on [LiDAR-Diffusion (LiDM)](https://github.com/hancyran/LiDAR-Diffusion).
