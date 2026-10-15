# Fine-Tuning the VGGT Depth Head on Driving Scenes

Can a large 3D foundation model get better at **depth estimation for autonomous driving** by retraining only a small part of it on **synthetic** driving data, and does that improvement carry over to **real** roads?

This project fine-tunes only the depth head of [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) on **Virtual KITTI 2** and evaluates it on both synthetic data and **real KITTI** images that the model never saw during training.

*Project Work in Image Processing and Computer Vision, University of Bologna. Supervisor: Prof. Giuseppe Lisanti.*

---

## Key results

All results on real KITTI use 192 test frames, 8-frame windows and median scaling.

| Model | Real KITTI AbsRel ↓ | Real KITTI δ<1.25 ↑ | Virtual KITTI AbsRel ↓ |
|---|---|---|---|
| Pretrained VGGT | 0.0489 | 0.9695 | 0.0384 |
| Fine-tuned (last epoch) | 0.0502 | 0.9675 | **0.0379** |
| **Fine-tuned (epoch selected on real validation drives)** | **0.0469** | **0.9719** | 0.0435 |

- **Fine-tuning on synthetic data alone does not transfer to real images.** The final model is slightly better on Virtual KITTI but worse on 72% of real-KITTI frames.
- **Real-image accuracy changes non-monotonically during training.** Choosing the epoch on separate real validation drives finds a model that is **4.1% better** than the pretrained one on the real test set (better on 73% of frames). The selected epoch is also best on the test drives, with a validation–test correlation of 0.74.
- **More views help a lot.** From 2 to 16 input views, the pretrained model's error drops by **23%** on real KITTI. The gain from fine-tuning is largest with few views (−7% at 2 views) and almost disappears at 16 (−1.3%).

The notebook also contains the per-frame comparisons, per-drive results and all figures.

---

## Method

VGGT takes several images of a scene and predicts, in one forward pass, the camera parameters, a depth map, a point map and tracking features for every image. It has a 1.2B-parameter transformer backbone (the *aggregator*) followed by small prediction heads.

In this project:
- the **aggregator is frozen**, and only the **DPT depth head** is trained;
- the camera, point and tracking heads are disabled;
- training uses the official VGGT training framework and its Virtual KITTI data loader.

| Setting | Value |
|---|---|
| Training data | Virtual KITTI 2, Scenes 01, 06, 18, 20 (`clone`, `Camera_0`) |
| Frames per sample | 2–8 (at most 8 images per GPU) |
| Image width | 518 px; aspect ratio (height/width) sampled in [0.28, 0.34] |
| Optimizer | AdamW, lr 5e-5, 5% warm-up + cosine decay, weight decay 0.05 |
| Training length | 1,000 steps (about 6 minutes on one A100) |
| Precision | bfloat16 on A100 (float16 on T4) |

---

## Data

| Dataset | Use | Frames |
|---|---|---|
| Virtual KITTI 2, Scenes 01/06/18/20 | Training | ~1,900 |
| Virtual KITTI 2, Scene02 | Synthetic test | 160 |
| Real KITTI, drives `2011_09_26_drive_0020/0036/0079` | Validation (epoch selection only) | 192 |
| Real KITTI, drives `2011_09_26_drive_0002/0005/0013` | Real test | 192 |

- Virtual KITTI 2 has exact depth for every pixel. It is also part of VGGT's original training data, so it is **not** a test of generalisation.
- Real KITTI uses the sparse LiDAR ground truth of the KITTI depth benchmark and is **not** in VGGT's training data.
- Only the required KITTI files are downloaded, read directly from inside the official zip archives with [`remotezip`](https://github.com/gtsystem/python-remotezip), a few hundred MB instead of ~190 GB.

---

## Evaluation protocol

The same code evaluates every model:
- **Multi-view:** consecutive frames are grouped into non-overlapping windows of *n* frames (default 8), and every frame is evaluated.
- **Resolution:** VGGT's standard preprocessing (width 518); predictions are resized back to the ground-truth resolution.
- **Scale:** VGGT predicts depth up to scale, so each frame is aligned with **median scaling**.
- **Valid pixels:** depth 0.001–80 m; for KITTI also the standard Garg crop.
- **Metrics:** AbsRel, SqRel, RMSE, RMSE log, δ<1.25, δ<1.25², δ<1.25³.
- **Sanity check:** before evaluating a fine-tuned model, the code verifies that every depth-head tensor was loaded and differs from the pretrained weights.

---

## Repository structure

```
├── VGGT_finetuning_clean.ipynb   # the full pipeline (Google Colab)
├── README.md
└── LICENSE
```

## Notebook overview

The notebook is designed for Google Colab. Everything that must survive a runtime restart (data, checkpoints, results, figures) is stored on Google Drive.

| Part | What it does | When to run |
|---|---|---|
| A | Setup: GPU, Drive, clone VGGT, dependencies | every session |
| B | Replication on the official example scene | once |
| C1 | Virtual KITTI: prepare train/test scenes (cached as one archive on Drive) | every session (fast after the first time) |
| C2 | Real KITTI: download the test drives | once (resumable) |
| D | Evaluation code | every session |
| E | Baseline evaluation of the pretrained model | once |
| F | Fine-tuning the depth head | once |
| G | Evaluation of the fine-tuned model, plus per-frame comparison | after F |
| I | Epoch selection on real validation drives, and final test | after F |
| H | View-count ablation (2/4/8/16 views) and summary table | last |

**A normal session:** A → C1 → C2 → D → the part you are working on.

---

## Reproducing the results

1. Open `VGGT_finetuning_clean.ipynb` in Google Colab and choose a GPU runtime (an A100 is fastest; a T4 works).
2. Put the Virtual KITTI 2 archives (`vkitti_2.0.3_rgb.tar`, `vkitti_2.0.3_depth.tar`, `vkitti_2.0.3_textgt.tar.gz`) in `MyDrive/vggt-depth-finetune/data/raw_tars/` on Google Drive. They are available from the [Virtual KITTI 2 website](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/).
3. Run the parts in the order shown in the table above. The real-KITTI data is downloaded automatically.

**Practical notes:**
- The T4 does not support bfloat16, so the notebook switches to float16 automatically.
- Do not run `pip install -e .` on the VGGT repo in Colab: it downgrades NumPy and breaks TensorBoard. The notebook adds the repo to `PYTHONPATH` instead.
- In VGGT's training code, the aspect ratio is **height / width**. KITTI frames are about 0.30.

---

## Limitations and future work

- The test set is small (3 drives, 192 frames), and each configuration was trained once.
- Median scaling evaluates the shape of the depth maps, not their metric scale.
- **Next step:** fine-tune on **real** KITTI, or on a mix of real and synthetic data. This needs a data loader that converts KITTI's GPS/IMU data into camera poses, and a depth loss adapted to sparse LiDAR ground truth.

---

## Acknowledgements and license

- VGGT: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025, [facebookresearch/vggt](https://github.com/facebookresearch/vggt). The VGGT code and model weights are © Meta Platforms and are subject to their own license. This repository does not redistribute them; they are downloaded from the official sources.
- Virtual KITTI 2: Cabon et al., 2020. KITTI: Geiger et al., 2013; depth benchmark: Uhrig et al., 2017.
- The code in this repository is released under the MIT License (see `LICENSE`).

```bibtex
@inproceedings{wang2025vggt,
  title     = {{VGGT}: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {CVPR},
  year      = {2025}
}
```
