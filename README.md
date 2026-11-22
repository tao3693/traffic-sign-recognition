# Traffic Sign Recognition with Deep Learning

**AIT202 Methods and Applications of Deep Learning — Group Project (Topic 7)**

Classify the 43 classes of the German Traffic Sign Recognition Benchmark (GTSRB) with two models and explain their decisions with Grad-CAM:

| Model | Description |
|-------|-------------|
| `cnn` | Custom CNN built from scratch (3 conv blocks + BatchNorm + Dropout, ~0.3 M params) — baseline |
| `resnet18` | ResNet18 pre-trained on ImageNet, fine-tuned on GTSRB (transfer learning) |

The program can train both models, evaluate them on the official test set, and classify your own images from a simple GUI or the command line.

> `README.txt` is the plain-text version required for the submission; this file is the GitHub homepage and also documents the team workflow.

---

## Contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Key design decisions](#key-design-decisions)
- [Running experiments](#running-experiments)
- [Outputs](#outputs)
- [Team workflow (PR process)](#team-workflow-pr-process)
- [Timeline](#timeline)
- [References](#references)

---

## Quick start

Requirements: Python 3.9+, a GPU is optional.

```bash
pip install -r requirements.txt
```

```bash
python main.py --download                 # download GTSRB (~350 MB) into data/GTSRB
```

```bash
python main.py --train                    # train the custom CNN
python main.py --train --model resnet18   # train ResNet18
```

```bash
python main.py                                   # GUI: choose an image, see top-3 prediction
python main.py --image my_sign.png               # prediction in the terminal
python main.py --image my_sign.png --gradcam     # + Grad-CAM heat-map in results/
```

```bash
python main.py --evaluate --weights weights/latest_cnn.pt   # final test-set evaluation
```

---

## Project structure

```
traffic-sign-recognition/
├── main.py              # entry point: GUI / predict / download / train / evaluate
├── config.yaml          # ALL hyper-parameters (overridable from the CLI)
├── download_data.py     # downloads + extracts the official GTSRB archives
├── requirements.txt
├── src/
│   ├── dataset.py       # CSV parsing, track-based split, ROI crop, augmentation, balanced sampler
│   ├── models/
│   │   ├── __init__.py  # build_model(cfg) factory
│   │   ├── cnn.py       # TrafficSignCNN (from scratch)
│   │   └── resnet.py    # ResNet18 transfer learning (optional frozen backbone)
│   ├── train.py         # training loop, grad clipping/monitoring, early stopping, experiment log
│   ├── evaluate.py      # test accuracy, per-class report, confusion matrix, error examples
│   ├── gradcam.py       # Grad-CAM implementation (hooks on the last conv layer)
│   ├── eda.py           # dataset statistics & figures for the report
│   └── utils.py         # config loading, seeding, device, class names, CLI flags
├── data/                # dataset (git-ignored)
├── weights/             # trained checkpoints (git-ignored, *.pt)
├── results/             # experiment logs, curves, test reports
└── report/figures/      # figures used in the written report
```

---

## Key design decisions

| Decision | Why | Where |
|----------|-----|-------|
| **Validation split by track**, not by image | GTSRB images come in tracks of ~30 consecutive frames of the same physical sign. A random split puts near-identical frames in train and val → inflated validation accuracy. We hold out whole tracks per class. | `src/dataset.py::split_by_track` |
| **Official test set used only once** | All tuning is done on the validation split; `evaluate.py` is run on the final models only, so the reported test accuracy is unbiased. | `src/evaluate.py` |
| **ROI crop** | The annotation CSV gives the sign's bounding box; cropping removes background clutter. | `crop_roi` in `config.yaml` |
| **No horizontal flip** | Flipping changes meaning ("turn left" ↔ "turn right", "keep left" ↔ "keep right"). Augmentation = small rotation, translation/scale, colour jitter. | `build_transforms` |
| **Class imbalance** (~210 → ~2250 images per class) | `WeightedRandomSampler` with weight = 1 / class frequency; label smoothing available as an experiment. | `make_balanced_sampler` |
| **GTSRB-specific normalisation** | Mean/std computed on GTSRB training images rather than ImageNet statistics. | `GTSRB_MEAN/STD` |
| **Input size** | CNN: 48×48 (signs are small, 15–250 px). ResNet18: 64×64 or larger so the deep stride-32 network keeps spatial resolution. | `img_size` |
| **Training stability** | Gradient-norm clipping (and logging), `ReduceLROnPlateau`, early stopping on val accuracy, fixed seed. | `src/train.py` |
| **Transfer-learning modes** | Full fine-tuning vs. frozen backbone (feature extraction) — compared as an experiment. | `freeze_backbone` |
| **Explainability** | Grad-CAM implemented from scratch on the last conv layer of each model to check that predictions rely on the sign, not the background. | `src/gradcam.py` |

---

## Running experiments

Every value in `config.yaml` can be overridden from the command line. Every run appends one row to `results/experiments.csv`, which becomes the results table in the report. Use `--tag` to give runs readable names.

```bash
python -m src.eda                                             # dataset statistics + figures
```

```bash
# --- CNN baseline & ablations ---
python -m src.train --tag baseline
python -m src.train --lr 0.01 --tag lr1e-2
python -m src.train --lr 0.0001 --tag lr1e-4
python -m src.train --optimizer sgd --lr 0.01 --tag sgd
python -m src.train --batch_size 128 --tag bs128
python -m src.train --augment false --tag noaug
python -m src.train --balanced_sampling false --tag nobalance
python -m src.train --label_smoothing 0.1 --tag ls0.1
python -m src.train --img_size 32 --tag img32
```

```bash
# --- ResNet18 transfer learning ---
python -m src.train --model resnet18 --img_size 64 --lr 0.0001 --tag finetune
python -m src.train --model resnet18 --img_size 64 --lr 0.001 --freeze_backbone true --tag frozen
python -m src.train --model resnet18 --img_size 96 --lr 0.0001 --tag finetune96
```

```bash
# --- final evaluation & explainability (after tuning only!) ---
python -m src.evaluate --weights weights/latest_cnn.pt
python -m src.evaluate --weights weights/latest_resnet18.pt
python -m src.gradcam --weights weights/latest_resnet18.pt --image path/to/sign.ppm
```

## Outputs

| File | Content |
|------|---------|
| `results/experiments.csv` | one summary row per run (hyper-parameters, best val acc, epoch, time) |
| `results/<exp>_history.csv` / `_curves.png` | per-epoch loss, accuracy, grad norm, lr |
| `results/<name>_test_report.txt` | test accuracy + per-class precision / recall / F1 |
| `results/<name>_confusion.png` / `_errors.png` | 43×43 confusion matrix, misclassified examples |
| `report/figures/*.png` | EDA figures (class distribution, samples, image sizes) |

---

## Team workflow (PR process)

| Role | Member | Responsibility |
|------|--------|----------------|
| A — Lead | @tao3693 | repo maintenance, code review & merging, progress reports, report integration |
| B — Data | TBD | dataset, EDA, data pipeline |
| C — CNN | TBD | custom CNN, baseline, hyper-parameter experiments |
| D — ResNet | TBD | literature review, ResNet18 transfer learning |
| E — Evaluation | TBD | metrics, confusion matrix, error analysis, Grad-CAM |

**Rules**

1. **Never push directly to `main`.** `main` must always run.
2. One task = one Issue = one branch: `feat/<role>-<short-topic>`, e.g. `feat/c-cnn-baseline`, `fix/b-csv-path`, `docs/d-related-work`.
   ```bash
   git checkout main && git pull
   git checkout -b feat/c-cnn-baseline
   ```
3. Commit small, with clear messages (`Add track-based split stats to eda.py`).
4. Open a Pull Request into `main`, write `Closes #<issue>` in the description, and attach results (val accuracy, curves, screenshots).
5. **At least one other member reviews** before merging; the lead merges (squash merge).
6. **Do not commit** the dataset, `*.pt` weights or zip files (already in `.gitignore`). Share weights via the group drive; commit only small result files (`.csv`, `.png`, `.txt`).
7. Change hyper-parameters through CLI flags or `config.yaml` in your PR — never hard-code them.
8. Record every experiment you want in the report in `results/experiments.csv` (automatic) and mention it in your PR.

---

## Timeline

| Date | Milestone |
|------|-----------|
| Tue 24 Nov | Progress check with lecturer: data + EDA, CNN baseline, ResNet plan, evaluation plan |
| 1 Dec | All experiments running; ResNet18 fine-tuning results |
| 6 Dec | Final evaluation on test set, Grad-CAM figures, report draft |
| 11 Dec | Report & code freeze, internal review |
| **13 Dec** | **Submission** |

---

## References

1. J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks 32, 323–332, 2012. — GTSRB dataset ([download](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html))
2. D. Cireşan, U. Meier, J. Masci, J. Schmidhuber. *Multi-column deep neural network for traffic sign classification.* Neural Networks 32, 333–338, 2012.
3. P. Sermanet, Y. LeCun. *Traffic sign recognition with multi-scale convolutional networks.* IJCNN 2011.
4. K. He, X. Zhang, S. Ren, J. Sun. *Deep residual learning for image recognition.* CVPR 2016. ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
5. S. Ioffe, C. Szegedy. *Batch normalization: Accelerating deep network training by reducing internal covariate shift.* ICML 2015.
6. N. Srivastava et al. *Dropout: A simple way to prevent neural networks from overfitting.* JMLR 15, 2014.
7. J. Yosinski, J. Clune, Y. Bengio, H. Lipson. *How transferable are features in deep neural networks?* NeurIPS 2014.
8. R. R. Selvaraju et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017. ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391))
9. D. P. Kingma, J. Ba. *Adam: A method for stochastic optimization.* ICLR 2015.
10. A. Paszke et al. *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS 2019.
