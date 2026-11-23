"""
Traffic Sign Recognition - AIT202 Group Project (Topic 7)
Entry point of the program.

    python main.py                              open the GUI (choose an image, see prediction)
    python main.py --image sign.png             predict one image in the terminal
    python main.py --image sign.png --gradcam   also save a Grad-CAM heat-map
    python main.py --download                   download the GTSRB dataset
    python main.py --train [--model resnet18]   train a model (settings in config.yaml;
                                                any src.train flag works, e.g. --lr 0.0001)
    python main.py --evaluate                   evaluate on the official test set

Images given to the GUI / --image should be cropped tightly around the sign, because the
models are trained on the ROI-cropped GTSRB images.
"""
import argparse
import os
import sys

from src.utils import add_common_args

_MODEL_CACHE = {}


def default_weights(model: str = None) -> str:
    return os.path.join("weights", f"best_{model or 'cnn'}.pt")


def get_model(weights: str):
    """Load a checkpoint once and reuse it (the GUI predicts many images)."""
    if weights not in _MODEL_CACHE:
        from src.evaluate import load_checkpoint
        from src.utils import get_device
        _MODEL_CACHE[weights] = load_checkpoint(weights, get_device())
    return _MODEL_CACHE[weights]


def predict(image_path: str, weights: str, topk: int = 3, gradcam: bool = False):
    import torch
    from PIL import Image

    from src.dataset import build_transforms
    from src.utils import CLASS_NAMES

    model, cfg = get_model(weights)
    device = next(model.parameters()).device
    tf = build_transforms(cfg["img_size"], train=False, normalize=cfg.get("normalize", "gtsrb"))
    x = tf(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0]
    top = probs.topk(topk)
    results = [(CLASS_NAMES[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]

    cam_path = None
    if gradcam:
        from src.gradcam import run_gradcam
        os.makedirs(cfg["results_dir"], exist_ok=True)
        cam_path = run_gradcam(model, cfg, image_path,
                               os.path.join(cfg["results_dir"], "gradcam_demo.png"))
    return results, cam_path


def launch_gui(weights: str):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Traffic Sign Recognition")
    root.geometry("420x480")
    img_label = tk.Label(root)
    img_label.pack(pady=10)
    result_var = tk.StringVar(
        value="Choose a traffic sign image.\n(Crop it tightly around the sign for best results.)")
    tk.Label(root, textvariable=result_var, justify="left", font=("Arial", 11)).pack(pady=10)

    def choose():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.ppm *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            results, _ = predict(path, weights)
        except Exception as exc:  # show any error to the user instead of crashing
            messagebox.showerror("Error", str(exc))
            return
        photo = ImageTk.PhotoImage(Image.open(path).convert("RGB").resize((200, 200)))
        img_label.configure(image=photo)
        img_label.image = photo
        result_var.set("\n".join(f"{name}: {p:.1%}" for name, p in results))

    tk.Button(root, text="Open image...", command=choose, width=20).pack()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition (GTSRB)")
    parser.add_argument("--image", help="image to classify")
    parser.add_argument("--weights", help="checkpoint (default: weights/best_<model>.pt)")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    add_common_args(parser)            # --model, --lr, --img_size, ... (used by --train)
    args = parser.parse_args()
    weights = args.weights or default_weights(args.model)

    if args.download:
        import download_data
        download_data.main()
    elif args.train:
        from src.train import train
        from src.utils import load_config
        overrides = {k: v for k, v in vars(args).items()
                     if k not in ("image", "weights", "gradcam", "download", "train",
                                  "evaluate", "config")}
        train(load_config(args.config, overrides))
    elif args.evaluate:
        sys.argv = ["evaluate", "--weights", weights]
        from src.evaluate import main as evaluate_main
        evaluate_main()
    elif args.image:
        if not os.path.exists(weights):
            sys.exit(f"Weights not found: {weights}. Train first or download them (see README).")
        results, cam = predict(args.image, weights, gradcam=args.gradcam)
        for name, p in results:
            print(f"{name:45s} {p:6.1%}")
        if cam:
            print("Grad-CAM saved to", cam)
    else:
        launch_gui(weights)


if __name__ == "__main__":
    main()
