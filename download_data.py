"""
Download the official GTSRB dataset (final training + final test + test ground truth).

Source: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
Citation: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, "Man vs. Computer: Benchmarking
Machine Learning Algorithms for Traffic Sign Recognition", Neural Networks 32, 2012.

Usage:
    python download_data.py
Result:
    data/GTSRB/Final_Training/Images/00000 ... 00042   (images + GT-xxxxx.csv)
    data/GTSRB/Final_Test/Images                     (test images)
    data/GTSRB/GT-final_test.csv                     (test labels)
"""
import os
import sys
import urllib.request
import zipfile

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
FILES = {
    # file name                          -> folder to extract into
    "GTSRB_Final_Training_Images.zip": "data",
    "GTSRB_Final_Test_Images.zip": "data",
    "GTSRB_Final_Test_GT.zip": os.path.join("data", "GTSRB"),
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done * 100 / total_size
    sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_and_extract(name: str, extract_to: str) -> None:
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join("data", name)
    if not os.path.exists(zip_path):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, zip_path, _progress)
        print()
    else:
        print(f"{name} already downloaded, skipping.")
    print(f"Extracting {name} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def main():
    os.makedirs("data", exist_ok=True)
    for name, target in FILES.items():
        download_and_extract(name, target)
    print("Done. Dataset is in data/GTSRB")


if __name__ == "__main__":
    main()
