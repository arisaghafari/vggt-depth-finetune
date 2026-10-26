# Fine-Tuning the VGGT Depth Head on Driving Scenes

Can a large 3D foundation model get better at **depth estimation for driving scenes** by retraining only a small part of it on **synthetic** data, and does that improvement carry over to **real** roads?

This project fine-tunes only the depth head of [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) on **Virtual KITTI 2** and tests it on **real KITTI** images that the model never saw during training. It does this at VGGT's normal input width (518 px) and at a lower width (336 px).

*Project Work in Image Processing and Computer Vision, University of Bologna. Supervisor: Prof. Giuseppe Lisanti.*

---

## Key results

Real KITTI: 192 test frames from 3 drives. Virtual KITTI: 160 frames of Scene02. All results use 8-frame windows and median scaling. AbsRel, lower is better.

| Model | Real KITTI | Virtual KITTI |
|---|---|---|
| Pretrained VGGT, width 518 | 0.0489 | 0.0384 |
| Fine-tuned, width 518, last epoch | 0.0481 | **0.0370** |
| **Fine-tuned, width 518, selected with early stopping** | **0.0453** (−7.4%) | 0.0404 |
| Pretrained VGGT, width 336 | 0.0897 | 0.0784 |
| Fine-tuned, width 336, last epoch | 0.0692 | **0.0618** |
| **Fine-tuned, width 336, selected with early stopping** | **0.0675** (−25%) | 0.0623 |

- **Early stopping on real validation drives finds a better model.** Real-image accuracy goes up and down during training. Choosing the epoch on 3 separate real-KITTI validation drives gives a model that is 7.4% better than the pretrained one on the real test set, and better on 80% of the frames. The validation and test curves agree closely (correlation 0.94).
- **At the lower resolution, fine-tuning helps much more.** At width 336, the pretrained model's error almost doubles. Fine-tuning lowers it by 25% and wins back about half of the lost accuracy, but it stays below the pretrained model at width 518.
- **More views help a lot.** From 2 to 16 views, the pretrained model's error on real KITTI drops by 23%. The gain from fine-tuning stays clear for every number of views (−8.2% with 2 views, −6.3% with 16).

---

## Method

VGGT takes several images of a scene and predicts, in one forward pass, the cameras, a depth map, a point map and tracking features for every image. Almost all of its 1.2B parameters are in a transformer backbone, the *aggregator*, followed by small prediction heads.

In this project:
- the **aggregator is frozen**, and only the **DPT depth head** is trained, with VGGT's depth loss;
- the camera, point and tracking heads are switched off;
- training uses the official VGGT training code and its Virtual KITTI data loader.

| Setting | Value |
|---|---|
| Training data | Virtual KITTI 2, Scenes 01, 06, 18, 20 (`clone`, `Camera_0`) |
| Image width | 518 px (Experiment 1), 336 px (Experiment 2) |
| Aspect ratio (height/width) | sampled in [0.28, 0.34], close to KITTI |
| Frames per sample | 2–8 (at most 8 images per GPU) |
| Optimizer | AdamW, lr 5e-5, 5% warm-up + cosine decay, weight decay 0.05 |
| Training length | 3,000 steps (30 epochs of 100 steps) |
| Precision | bfloat16, one A100 GPU on Google Colab |

**Epoch selection.** After every epoch, the depth head is saved. The pretrained head and all 30 saved heads are evaluated on the validation drives, and early stopping (patience 5) chooses one. Only the selected head and the last one are evaluated on the test sets.

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
- Only the needed KITTI files are downloaded, read directly from inside the official zip archives with [`remotezip`](https://github.com/gtsystem/python-remotezip): a few hundred MB instead of ~190 GB.

---

## Evaluation protocol

The same code evaluates every model:
- **Multi-view:** consecutive frames are grouped into non-overlapping windows of *n* frames (default 8), and every frame is evaluated once.
- **Feed-forward only:** no bundle adjustment.
- **Scale:** VGGT predicts depth up to scale, so each frame is aligned with **median scaling**.
- **Valid pixels:** depth 0.001–80 m; for KITTI, also the standard Garg crop.
- **Metrics:** AbsRel, SqRel, RMSE, RMSE log, δ<1.25, δ<1.25², δ<1.25³, plus frame-by-frame comparisons with the pretrained model.

---

## Repository structure

```
├── VGGT_finetuning.ipynb   # the full pipeline (Google Colab)
├── report/                 # project report (PDF)
├── figures/                # figures used in the report
├── README.md
└── LICENSE
```

## Notebook overview

The notebook runs on Google Colab. Everything that must survive a runtime restart (data, depth heads, results, figures, logs) is stored on Google Drive.

| Part | What it does | When to run |
|---|---|---|
| A | Setup: GPU, Drive, VGGT code, dependencies | every session |
| B | Replication on the official example scene | once |
| C | Data: Virtual KITTI 2 and the real-KITTI test and validation drives | every session (fast after the first time) |
| D | Evaluation code | every session |
| E | Baseline: the pretrained model at width 518 | once |
| F | Training and selection tools (config, early stopping) | every session before G, I or K |
| G | **Experiment 1:** fine-tuning at width 518 | once |
| H | View-count ablation (2/4/8/16 views) | after G |
| I | **Experiment 2:** fine-tuning at width 336 | once |
| K | Extra tests: averaging, saving more often, smaller learning rate | optional |
| L | Summary table of all results | last |

**A normal session:** A → C → D → F → the part you are working on.

---

## Reproducing the results

1. Open `VGGT_finetuning.ipynb` in Google Colab with an **A100** (or L4) runtime. Training needs bfloat16, which older GPUs such as the T4 do not support.
2. Download the Virtual KITTI 2 archives from the [Virtual KITTI 2 website](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/), rename them to `vkitti_rgb.tar`, `vkitti_depth.tar` and `vkitti_textgt.tar.gz`, and put them in `MyDrive/vggt-depth-finetune/data/raw_tars/` on Google Drive.
3. Run the parts in the order of the table above. The real-KITTI data is downloaded automatically.

**Practical notes:**
- Each training run takes roughly 30 minutes on an A100. The depth head of every epoch is kept on Drive, so the selection and the tests can be repeated without training again.
- Do not run `pip install -e .` on the VGGT repo in Colab: it downgrades NumPy and breaks other libraries. The notebook adds the repo to `PYTHONPATH` instead.
- In VGGT's training code, the aspect ratio is **height / width**. KITTI frames are about 0.30.

---

## Limitations and future work

- The test set is small (3 drives, 192 frames), and each configuration was trained once, with one random seed.
- Median scaling evaluates the shape of the depth maps, not their metric scale.
- **Next step:** fine-tune on **real** KITTI, or on a mix of real and synthetic data. This needs a data loader that turns KITTI's GPS/IMU data into camera poses, and a depth loss for sparse LiDAR ground truth.

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