import csv
import os
import statistics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image

from rex_xai.lib import ReX
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction

from src.data import targets_of
from src.utils.naming import results_location
from src.utils.device import get_device, amp_dtype, autocast
from src.config import Config, MEAN, STD
cfg = Config()


def tensor_to_pil(x, mean=MEAN, std=STD):
    # x: [3, H, W], normalised with the same MEAN/STD the data loaders use
    x = x.detach().cpu().clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = (x * std + mean).clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def stratified_subset(dataset, per_class, seed=0):
    """The same `per_class` images of every class, whatever checkpoint is measured.

    Using one fixed set across checkpoints means changes in explanation size
    are paired comparisons on the same images, not different random samples.
    """
    if per_class is None:
        return dataset
    labels = targets_of(dataset)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    g = torch.Generator().manual_seed(seed)
    chosen = []
    for y in sorted(by_class):
        idx = by_class[y]
        perm = torch.randperm(len(idx), generator=g)[:per_class].tolist()
        chosen.extend(sorted(idx[p] for p in perm))
    return Subset(dataset, chosen)


def make_prediction_function(model, device, dtype=None):
    """A faster drop-in for ReX's default prediction function.

    ReX's default (rex_xai/responsibility/prediction.py) runs in fp32 and calls
    .item() several times per occluded image, and every .item() waits for the
    GPU to finish. This does softmax and top-1 on the GPU and copies the
    results to the CPU once per batch. With dtype=None the numbers are the
    same as the default's; bf16/fp16 is faster but can flip borderline
    predictions, so compare against fp32 on a few images before switching.
    """
    def inner(mutants, target=None, raw=False):
        with torch.no_grad(), autocast(device, dtype):
            logits = model(mutants)
        probs = F.softmax(logits.float(), dim=1)
        if raw:
            return probs

        conf, cls = probs.max(dim=1)
        conf, cls = conf.tolist(), cls.tolist()
        target_conf = probs[:, target.classification].tolist() if target is not None else None

        preds = []
        for i in range(len(cls)):
            p = Prediction(cls[i], conf[i])
            if target is not None:
                # Mirrors the default function exactly, including storing the
                # target Prediction object on p.target.
                p.target = target
                p.target_confidence = target_conf[i]
            preds.append(p)
        return preds

    return inner


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)  # a crash never leaves a half-written results file


def per_class_path(epoch, model_name=cfg.model):
    return os.path.join(results_location(model_name), f"per_class_epoch_{int(epoch):03d}.csv")


def per_image_path(epoch, model_name=cfg.model):
    return os.path.join(results_location(model_name), f"per_image_epoch_{int(epoch):03d}.csv")


def measure(model, data, idx="", precision="fp32"):
    """Run ReX on every image in `data` and write two CSVs for checkpoint `idx`:
    one row per image, and the per-class summary (mean, stdev, accuracy).
    Both files are overwritten, so rerunning a checkpoint never duplicates rows.
    """
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    device = get_device()
    dtype = amp_dtype(device, precision)
    model.eval().to(device)
    print(f"Measuring checkpoint {idx} on {len(data)} images, device={device}, autocast={dtype}")

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    rex = ReX(model, ["N", 3, 224, 224], device, "RGB",
              prediction_function=make_prediction_function(model, device, dtype))
    rex.args.minimum_confidence_threshold = 0.5
    rex.args.batch_size = cfg.rex_batch_size
    rex.args.iters = cfg.rex_iters
    rex.args.seed = 0

    rows = []
    correct = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)  # [1, 3, 224, 224]
        x0 = x.squeeze(0)

        d = Data(
            input=tensor_to_pil(x0),
            model_shape=["N", 3, 224, 224],
            device=device,
            mode="RGB",
            process=False,
        )
        d.data = x
        rex.data = d

        rex.set_target()

        pred_class = int(rex.data.target.classification)
        true_class = int(y.item())
        is_correct = pred_class == true_class
        correct += int(is_correct)

        rex.calculate_responsibility()
        rex.generate_explanation_object()
        rex.extract_sufficient_explanation()

        mask = rex.explanation.sufficiency_mask
        size = torch.count_nonzero(mask).item() / mask.numel()

        rows.append({
            "idx": idx,
            "image": i,
            "true_class": true_class,
            "pred_class": pred_class,
            "correct": int(is_correct),
            "confidence": float(rex.data.target.confidence),
            "size": size,
        })

        if (i + 1) % 50 == 0:
            sizes = [r["size"] for r in rows]
            print(f"[{i + 1}/{len(data)}] mean={statistics.fmean(sizes):.4f} "
                  f"accuracy={correct / len(rows):.4f}")

    _write_csv(per_image_path(idx), rows)

    summary = []
    for cls in sorted({r["true_class"] for r in rows}):
        cls_rows = [r for r in rows if r["true_class"] == cls]
        cls_sizes = [r["size"] for r in cls_rows]
        cls_correct = sum(r["correct"] for r in cls_rows)
        summary.append({
            "idx": idx,
            "class": cls,
            "mean": statistics.fmean(cls_sizes),
            "stdev": statistics.stdev(cls_sizes) if len(cls_sizes) > 1 else 0.0,
            "accuracy": cls_correct / len(cls_rows),
            "correct": cls_correct,
            "total": len(cls_rows),
        })
    _write_csv(per_class_path(idx), summary)

    return rows
