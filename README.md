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

Requirements: Python 3.9+, a GPU is optional (training on CPU works but is slower).

```bash
pip install -r requirements.txt
```

**GPU training (recommended):** the code uses an NVIDIA GPU automatically when one is available, with mixed precision (`amp: true` in `config.yaml`). Install a CUDA build of PyTorch using the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For maximum speed (not bit-for-bit reproducible) set `deterministic: false` in `config.yaml`, which enables `cudnn.benchmark`.

```bash
python main.py --download                 # download GTSRB (~350 MB) into data/GTSRB
```

```bash
python main.py --train                    # train the custom CNN (48x48)
python main.py --train --model resnet18   # train ResNet18 (96x96 by default)
```

```bash
python main.py                                   # GUI: choose an image, see top-3 prediction
python main.py --image my_sign.png               # prediction in the terminal
python main.py --image my_sign.png --gradcam     # + Grad-CAM heat-map in results/
python main.py --image my_sign.png --model resnet18
```

> Use images **cropped tightly around the sign** — the models are trained on ROI-cropped GTSRB images, so a full street scene will be classified poorly.

```bash
python main.py --evaluate --weights weights/best_cnn.pt     # final test-set evaluation
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
│   ├── eda.py           # dataset statistics, mean/std & figures for the report
│   ├── summary.py       # per-run JSON records -> experiments.csv / test_results.csv
│   └── utils.py         # config, seeding, device, experiment names, CLI flags
├── tests/test_smoke.py  # end-to-end smoke tests on a fake mini-GTSRB (pytest -q)
├── .github/             # CI workflow, PR template, issue templates
├── data/                # dataset (git-ignored)
├── weights/             # checkpoints (git-ignored): <experiment>.pt, best_cnn.pt, best_resnet18.pt
├── results/runs/        # one JSON per training run (committed)
├── results/tests/       # one JSON per test-set evaluation (committed)
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
| **Normalisation** | GTSRB mean/std by default (verify with `python -m src.eda --stats`); `--normalize imagenet` uses the statistics the pre-trained ResNet expects — compared as an experiment. | `normalize` |
| **Input size** | CNN: 48×48 (signs are small, 15–250 px). ResNet18 down-samples 32×, so it uses 96×96 by default (`resnet_img_size`); `small_input` replaces the 7×7/stride-2 stem + max-pool with a 3×3/stride-1 conv (8× down-sampling) for small images. | `img_size`, `resnet_img_size`, `small_input` |
| **Training stability & reproducibility** | Mixed-precision (AMP) training on GPU with gradient scaling; gradient-norm clipping on the unscaled gradients (and logging), `ReduceLROnPlateau`, early stopping on val accuracy; fixed seed for Python/NumPy/torch, DataLoader generator & workers, cuDNN deterministic mode; python/torch/device logged per run. | `src/train.py`, `src/utils.py` |
| **Experiment bookkeeping** | The experiment name encodes every varied setting (img size, optimizer, lr, bs, weight decay, momentum, aug, sampling, smoothing, dropout, normalisation, ResNet mode) plus the start time, so runs never overwrite each other — not even across team members' machines. Each run writes its own `results/runs/<name>.json`; the CSV tables are generated from these files, so parallel PRs never conflict. | `experiment_name`, `src/summary.py` |
| **Fair model selection** | `best_<model>.pt` is only replaced by a run with higher val accuracy **on the same validation split** (same `val_ratio`, `seed`, `crop_roi`). It is a local convenience; the team chooses the final model by experiment name. | `update_best` |
| **Transfer-learning modes** | Full fine-tuning vs. frozen backbone (feature extraction) — compared as an experiment. When frozen, the backbone's BatchNorm layers are kept in eval mode too, so their ImageNet running statistics are not updated by GTSRB batches. | `freeze_backbone`, `set_train_mode` |
| **Explainability** | Grad-CAM implemented from scratch to check that predictions rely on the sign, not the background. Target layer is chosen for useful resolution: CNN last ReLU (12×12 at 48 px); ResNet18 `layer3` (6×6 at 96 px) or `layer4` with `small_input` (12×12) — standard `layer4` would only be 3×3. | `src/gradcam.py`, `get_gradcam_layer` |

---

## Running experiments

Every value in `config.yaml` can be overridden from the command line. Every run writes `results/runs/<experiment>.json` and regenerates `results/experiments.csv` (the results table for the report). After pulling teammates' results, run `python -m src.summary` to rebuild the tables. Use `--tag` to give runs readable names.

> ResNet18 input size: `--model resnet18` uses `resnet_img_size` (96) unless you pass `--img_size`; the `img_size` key is for the CNN.

```bash
python -m src.eda                                             # dataset statistics + figures
python -m src.eda --stats                                     # + per-channel mean / std
pytest -q                                                     # smoke tests (no dataset needed)
```

```bash
# --- CNN baseline & ablations ---
python -m src.train --tag baseline
python -m src.train --lr 0.01
python -m src.train --lr 0.0001
python -m src.train --optimizer sgd --lr 0.01
python -m src.train --batch_size 128
python -m src.train --augment false
python -m src.train --balanced_sampling false
python -m src.train --label_smoothing 0.1
python -m src.train --weight_decay 0.001
python -m src.train --img_size 32
python -m src.train --dropout 0.5
```

```bash
# --- ResNet18 transfer learning ---
python -m src.train --model resnet18 --lr 0.0001                          # fine-tune, 96x96
python -m src.train --model resnet18 --lr 0.001 --freeze_backbone true    # feature extraction
python -m src.train --model resnet18 --lr 0.0001 --img_size 64            # input size
python -m src.train --model resnet18 --lr 0.0001 --small_input true --img_size 64
python -m src.train --model resnet18 --lr 0.0001 --normalize imagenet
```

```bash
# --- final evaluation & explainability (after tuning only!) ---
python -m src.evaluate --weights weights/best_cnn.pt
python -m src.evaluate --weights weights/best_resnet18.pt
python -m src.gradcam --weights weights/best_cnn.pt --image path/to/sign.ppm
python -m src.gradcam --weights weights/best_resnet18.pt --image path/to/sign.ppm --layer layer3
```

## Outputs

| File | Content |
|------|---------|
| `results/runs/<exp>.json` | summary of one run: all hyper-parameters incl. weight decay, epochs, grad clip, val split; params, best val acc, epoch, time, python/torch/device — **commit it** |
| `results/experiments.csv` | table of all runs, generated from `results/runs/` (git-ignored; `python -m src.summary`) |
| `results/<exp>_config.yaml` | full configuration of the run — every setting, even those not in the name |
| `results/<exp>_history.csv` / `_curves.png` | per-epoch loss, accuracy, grad norm, lr |
| `results/<exp>_test_report.txt` | test accuracy, top-3 accuracy, macro-F1, params, ms/image, most-confused pairs, per-class P/R/F1 |
| `results/tests/<exp>.json` | summary of one test-set evaluation — **commit it** |
| `results/test_results.csv` | table of all evaluations, generated from `results/tests/` (git-ignored) |
| `results/<exp>_confusion.png` / `_errors.png` | 43×43 confusion matrix, misclassified examples |
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
4. Open a Pull Request into `main` (the PR template asks for `Closes #<issue>`, results and a checklist) and attach curves / screenshots.
5. **At least one other member reviews** before merging; the lead merges (squash merge). Track progress on the **Projects** board.
6. **CI must be green**: GitHub Actions runs `pytest -q` on every PR. Run it locally before pushing.
7. **Do not commit** the dataset, `*.pt` weights or zip files (already in `.gitignore`). Share weights via the group drive; commit only small result files (`results/runs/*.json`, `results/tests/*.json`, `*_config.yaml`, `*_history.csv`, `.png`, `.txt`). `experiments.csv` / `test_results.csv` are generated — don't commit them.
8. Change hyper-parameters through CLI flags or `config.yaml` in your PR — never hard-code them.
9. Mention every experiment you want in the report by its **experiment name** in your PR (its `results/runs/<name>.json` is the record).
10. **Final models are chosen by the team, by experiment name** — compare runs in `experiments.csv` that share the same validation split (`val_ratio`, `seed`, `crop_roi`). The lead uploads the chosen checkpoints to the shared drive as `best_cnn.pt` / `best_resnet18.pt`; everyone evaluates and demos with those files. Your local `weights/best_*.pt` is only your own best run.

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
