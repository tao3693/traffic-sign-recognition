# Traffic Sign Recognition with Deep Learning

> **AIT202 Methods and Applications of Deep Learning** — Group Project (Topic 7)

The program recognises **43 classes of German traffic signs** from an image using the
[GTSRB](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) benchmark.
Two models are compared:

| Model | Description |
|---|---|
| `cnn` | Convolutional neural network built from scratch (baseline) |
| `resnet18` | ResNet18 pre-trained on ImageNet, fine-tuned on GTSRB (transfer learning) |

You can train the models, evaluate them on the official test set, and classify your own images
through a simple GUI or the command line. **Grad-CAM** heat-maps show which part of the image the model looked at.

---

## Requirements

- Python 3.9 or newer
- A GPU is optional (training on CPU works but is slower)

```bash
pip install -r requirements.txt
```

**GPU training (recommended):** the code uses an NVIDIA GPU automatically when one is available,
with mixed precision (`amp: true` in `config.yaml`). Install a CUDA build of PyTorch using the selector at
[pytorch.org](https://pytorch.org/get-started/locally/), then check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick Start

```bash
# 1. Download the dataset (~350 MB, saved in data/)
python main.py --download

# 2. Train a model
python main.py --train                     # custom CNN
python main.py --train --model resnet18    # ResNet18 transfer learning

# 3. Use the program
python main.py                             # open the GUI
python main.py --image my_sign.png         # prediction in the terminal
python main.py --image my_sign.png --gradcam

# 4. Evaluate on the official test set
python main.py --evaluate --weights weights/latest_cnn.pt
```

## Project Structure

```
traffic-sign-recognition/
├── main.py              # program entry point
├── config.yaml          # all hyper-parameters
├── download_data.py     # downloads the GTSRB dataset
├── requirements.txt
├── README.txt           # instruction manual (submission version)
├── src/
│   ├── dataset.py       # data loading, track-based split, augmentation, sampling
│   ├── models/
│   │   ├── cnn.py       # custom CNN
│   │   └── resnet.py    # ResNet18 transfer learning
│   ├── train.py         # training loop, loss curves, experiment log
│   ├── evaluate.py      # test accuracy, classification report, confusion matrix
│   ├── gradcam.py       # Grad-CAM visualisation
│   ├── eda.py           # dataset statistics & figures for the report
│   └── utils.py         # config, seeding, class names
├── results/             # experiment logs and figures
├── report/figures/      # figures used in the report
├── weights/             # trained models (not committed)
└── data/                # dataset (not committed)
```

## Key Design Decisions

- **Track-based validation split** — each physical sign appears as a track of ~30 frames.
  Whole tracks are held out for validation, so near-identical frames never leak between train and val.
- **ROI cropping** — the sign is cropped using the bounding box from the annotation CSV.
- **No horizontal flip** — flipping changes the meaning of some signs (e.g. *turn left* ↔ *turn right*).
- **Class-balanced sampling** — `WeightedRandomSampler` (weight = 1 / class frequency) handles the
  imbalance between classes (~210 to ~2250 images per class).
- **Gradient clipping & monitoring** — gradient norms are clipped and logged every epoch.

## Running Experiments

All settings live in `config.yaml`; any value can be overridden from the command line:

```bash
python -m src.train --lr 0.01
python -m src.train --optimizer sgd --lr 0.01
python -m src.train --augment false --tag noaug
python -m src.train --model resnet18 --img_size 64 --freeze_backbone true
```

Each run produces:

| Output | Content |
|---|---|
| `results/experiments.csv` | one summary row per run (for the report's results table) |
| `results/<run>_history.csv` | per-epoch loss, accuracy, gradient norm, learning rate |
| `results/<run>_curves.png` | loss and accuracy curves |
| `weights/<run>.pt` | best checkpoint (by validation accuracy) |

Dataset figures for the report: `python -m src.eda` → `report/figures/`.

## Team Workflow

1. Never push directly to `main` — create a branch: `git checkout -b feature/<your-task>`
2. Commit often with clear messages, then open a **Pull Request**.
3. At least one teammate reviews before merging.
4. Track tasks with **Issues** and the **Projects** board.

## Dataset & References

- J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, "Man vs. Computer: Benchmarking Machine Learning
  Algorithms for Traffic Sign Recognition," *Neural Networks*, vol. 32, pp. 323–332, 2012.
- K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016.
- R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization," ICCV 2017.
