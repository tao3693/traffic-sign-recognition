"""
Smoke tests: run with `pytest -q` (no dataset or GPU needed, ~30 s on CPU).
A tiny fake GTSRB folder is generated so the whole pipeline (train -> evaluate ->
Grad-CAM -> predict) is exercised end to end.
"""
import csv
import os
import sys

import pytest
import torch
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import Sample, split_by_track                       # noqa: E402
from src.gradcam import GradCAM                                      # noqa: E402
from src.models import build_model, get_gradcam_layer, set_train_mode  # noqa: E402
from src.utils import NUM_CLASSES, experiment_name, load_config      # noqa: E402

FIELDS = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]


def _cfg(**kw):
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    cfg.update(pretrained=False, **kw)       # no ImageNet download in tests
    return cfg


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("model,size,small,cam", [
    ("cnn", 48, False, 12),
    ("resnet18", 96, False, 6),     # layer3, down-sampling 16
    ("resnet18", 96, True, 12),     # small_input -> layer4, down-sampling 8
])
def test_forward_and_gradcam_resolution(model, size, small, cam):
    cfg = _cfg(model=model, img_size=size, small_input=small)
    net = build_model(cfg)
    x = torch.randn(2, 3, size, size)
    assert net(x).shape == (2, NUM_CLASSES)
    with GradCAM(net, get_gradcam_layer(net, cfg)) as gc:
        heat, cls, prob, cam_size = gc(x[:1])
    assert cam_size == (cam, cam)
    assert heat.shape == (size, size) and 0 <= prob <= 1 and 0 <= cls < NUM_CLASSES
    assert not any(k.startswith("gradcam_layer") for k in net.state_dict())


def test_gradcam_frozen_backbone_and_hooks_removed():
    cfg = _cfg(model="resnet18", img_size=64, freeze_backbone=True)
    net = build_model(cfg)
    layer = get_gradcam_layer(net, cfg)
    with GradCAM(net, layer) as gc:
        gc(torch.randn(1, 3, 64, 64))
    assert not layer._forward_hooks and not layer._backward_hooks


@pytest.mark.parametrize("small", [False, True])
def test_frozen_backbone_batchnorm_statistics_do_not_change(small):
    cfg = _cfg(model="resnet18", img_size=64, freeze_backbone=True, small_input=small)
    net = build_model(cfg)
    frozen_bn = net.layer1[0].bn1
    before = frozen_bn.running_mean.clone()
    set_train_mode(net)
    net(torch.randn(4, 3, 64, 64) * 3 + 1)
    assert torch.equal(before, frozen_bn.running_mean)          # frozen BN untouched
    assert net.fc.training and not frozen_bn.training
    assert net.bn1.training == small                             # new stem BN is trainable
    assert all(p.requires_grad for p in net.bn1.parameters()) == small


# ------------------------------------------------------------------ config / naming
def test_resnet_uses_resnet_img_size_unless_overridden():
    path = os.path.join(ROOT, "config.yaml")
    assert load_config(path, {"model": "resnet18"})["img_size"] == 96
    assert load_config(path, {"model": "resnet18", "img_size": 64})["img_size"] == 64
    assert load_config(path, {"model": "cnn"})["img_size"] == 48


def test_summary_tables_union_columns(tmp_path):
    from src.summary import RUNS, load_records, rebuild, save_record
    save_record(str(tmp_path), RUNS, "a", {"experiment": "a", "started": "1", "lr": 0.1})
    save_record(str(tmp_path), RUNS, "b", {"experiment": "b", "started": "2", "wd": 0.01})
    path = rebuild(str(tmp_path), RUNS)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert [r["experiment"] for r in rows] == ["a", "b"] and rows[0]["wd"] == ""
    assert len(load_records(str(tmp_path), RUNS)) == 2


def test_experiment_names_differ_for_ablations():
    base = _cfg()
    names = {experiment_name(dict(base, **kw)) for kw in [
        {}, {"augment": False}, {"balanced_sampling": False}, {"label_smoothing": 0.1},
        {"img_size": 32}, {"dropout": 0.5}, {"normalize": "imagenet"},
        {"weight_decay": 1e-3}, {"optimizer": "sgd", "momentum": 0.8}, {"optimizer": "sgd"}]}
    assert len(names) == 10


def test_split_by_track_has_no_leakage():
    samples = [Sample(f"{c}_{t}_{i}", (0, 0, 1, 1), c, t)
               for c in range(3) for t in range(10) for i in range(5)]
    train, val = split_by_track(samples, 0.2, seed=0)
    tr = {(s.label, s.track) for s in train}
    va = {(s.label, s.track) for s in val}
    assert tr.isdisjoint(va) and len(train) + len(val) == len(samples)


# ------------------------------------------------------------------ end to end
def _make_fake_gtsrb(root, tracks=2, per_track=3, size=40):
    train_root = os.path.join(root, "Final_Training", "Images")
    for c in range(NUM_CLASSES):
        folder = os.path.join(train_root, f"{c:05d}")
        os.makedirs(folder)
        rows = []
        for t in range(tracks):
            for i in range(per_track):
                name = f"{t:05d}_{i:05d}.ppm"
                Image.new("RGB", (size, size), (c * 5, 100, 200 - c * 4)).save(
                    os.path.join(folder, name))
                rows.append([name, size, size, 5, 5, size - 5, size - 5, c])
        with open(os.path.join(folder, f"GT-{c:05d}.csv"), "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(FIELDS)
            w.writerows(rows)
    test_dir = os.path.join(root, "Final_Test", "Images")
    os.makedirs(test_dir)
    with open(os.path.join(root, "GT-final_test.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(FIELDS)
        for c in range(NUM_CLASSES):
            name = f"{c:05d}.ppm"
            Image.new("RGB", (size, size), (c * 5, 100, 200 - c * 4)).save(
                os.path.join(test_dir, name))
            w.writerow([name, size, size, 5, 5, size - 5, size - 5, c])
    return os.path.join(test_dir, "00000.ppm")


def test_train_evaluate_gradcam_predict(tmp_path, monkeypatch):
    image = _make_fake_gtsrb(str(tmp_path / "GTSRB"))
    cfg = _cfg()
    cfg.update(data_dir=str(tmp_path / "GTSRB"), results_dir=str(tmp_path / "results"),
               weights_dir=str(tmp_path / "weights"), epochs=1, batch_size=32,
               num_workers=0, img_size=32)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    from src.train import train
    ckpt = train(dict(cfg))
    best = os.path.join(cfg["weights_dir"], "best_cnn.pt")
    assert os.path.exists(ckpt) and os.path.exists(best)
    ckpt2 = train(dict(cfg))                       # same settings -> must not overwrite
    assert ckpt2 != ckpt
    name = os.path.splitext(os.path.basename(ckpt))[0]
    runs = os.path.join(cfg["results_dir"], "runs")
    assert sorted(os.listdir(runs)) == sorted(
        os.path.splitext(os.path.basename(c))[0] + ".json" for c in (ckpt, ckpt2))
    with open(os.path.join(cfg["results_dir"], "experiments.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["weight_decay"] == str(cfg["weight_decay"])

    # a run on a different validation split must not replace best_cnn.pt
    from src.train import update_best
    best_before = torch.load(best, weights_only=False)["experiment"]
    assert not update_best(ckpt2, 1.0, dict(cfg, seed=7))
    assert torch.load(best, weights_only=False)["experiment"] == best_before
    with open(os.path.join(cfg["results_dir"], f"{name}_config.yaml")) as f:
        saved = yaml.safe_load(f)
    assert saved["weight_decay"] == cfg["weight_decay"] and saved["experiment"] == name

    from src import evaluate
    monkeypatch.setattr(sys, "argv", ["evaluate", "--weights", best])
    evaluate.main()
    assert os.path.exists(os.path.join(cfg["results_dir"], f"{best_before}_test_report.txt"))
    assert os.path.exists(os.path.join(cfg["results_dir"], "tests", f"{best_before}.json"))
    assert os.path.exists(os.path.join(cfg["results_dir"], "test_results.csv"))

    from src.gradcam import run_gradcam
    model, loaded_cfg = evaluate.load_checkpoint(best, torch.device("cpu"))
    out = run_gradcam(model, loaded_cfg, image, str(tmp_path / "cam.png"))
    assert os.path.exists(out)

    import main
    results, _ = main.predict(image, best)
    assert len(results) == 3 and abs(sum(p for _, p in results)) <= 1.0001
