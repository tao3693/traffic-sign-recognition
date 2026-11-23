==============================================================
 Traffic Sign Recognition with Deep Learning
 AIT202 Methods and Applications of Deep Learning - Group Project
==============================================================

1. WHAT THE PROGRAM DOES
------------------------
The program recognises 43 classes of German traffic signs from an image.
Two models are provided:
  * cnn       - a convolutional neural network built from scratch
  * resnet18  - ResNet18 pre-trained on ImageNet, fine-tuned on GTSRB
You can train the models, evaluate them on the official test set, and
classify your own images through a simple GUI or the command line.
Grad-CAM heat-maps show which part of the image the model looked at.

2. REQUIREMENTS
---------------
  * Python 3.9 or newer
  * Packages:  pip install -r requirements.txt
  * A GPU is optional (training on CPU works but is slower)

3. QUICK START
--------------
  Step 1  Install packages
            pip install -r requirements.txt
  Step 2  Download the dataset (about 350 MB, saved in data/)
            python main.py --download
  Step 3  Train a model (or put the provided weights file in weights/)
            python main.py --train                 (custom CNN, 48x48 input)
            python main.py --train --model resnet18   (ResNet18, 96x96 input)
          The best model of each type is kept as weights/best_cnn.pt and
          weights/best_resnet18.pt (only replaced when a new run is better
          on the same validation split).
  Step 4  Use the program
            python main.py                         (opens the GUI)
            python main.py --image my_sign.png     (prediction in terminal)
            python main.py --image my_sign.png --gradcam
            python main.py --image my_sign.png --model resnet18
          NOTE: the models are trained on images cropped around the sign, so
          use images that are cropped tightly around the sign.
  Step 5  Evaluate on the official test set
            python main.py --evaluate --weights weights/best_cnn.pt

4. PROJECT STRUCTURE
--------------------
  main.py            program entry point
  config.yaml        all hyper-parameters
  download_data.py   downloads the GTSRB dataset
  src/dataset.py     data loading, track-based split, augmentation, sampling
  src/models/        cnn.py (custom CNN), resnet.py (ResNet18)
  src/train.py       training loop, loss curves, experiment log
  src/evaluate.py    test accuracy, classification report, confusion matrix
  src/gradcam.py     Grad-CAM visualisation
  src/eda.py         dataset statistics and figures for the report
  src/summary.py     builds the result tables from results/runs/*.json
  tests/             smoke tests (run: pytest -q)
  results/           experiment logs and figures
  weights/           trained model files

5. RUNNING EXPERIMENTS (hyper-parameter tuning)
-----------------------------------------------
  python -m src.train --lr 0.01
  python -m src.train --optimizer sgd --lr 0.01
  python -m src.train --augment false --tag noaug
  python -m src.train --model resnet18 --freeze_backbone true
  python -m src.train --model resnet18 --small_input true --img_size 64
Every run saves results/runs/<experiment>.json and updates the table
results/experiments.csv (rebuild it with: python -m src.summary).
The experiment name encodes the settings and the start time, so runs
never overwrite each other.

6. DATASET
----------
German Traffic Sign Recognition Benchmark (GTSRB)
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
J. Stallkamp et al., Neural Networks 32, pp. 323-332, 2012.
