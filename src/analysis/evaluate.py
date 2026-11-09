"""Measure how explainable a checkpoint is, using ReX.

For each test image, ReX finds a *sufficient explanation*: the smallest set of
pixels that, kept on their own (everything else masked to 0), still makes the
model give the same prediction. The explanation size is that set's share of
the image. The hypothesis is that adding a junk class makes these sets bigger,
which means the model is less explainable.

Outputs, under results/<model>/<JOB_ID>/:
  per_class.csv  one row per (checkpoint, class): mean/stdev size, accuracy,
                 plus an "overall" row per checkpoint (used by visualize.py)
  per_image.csv  one row per explained image: size, true/pred class,
                 softmax entropy and junk-class probability
"""

import csv
import functools
import math
import os
import statistics

import rex_xai.explanation.explanation as _rex_explanation
import torch
from rex_xai.input.input_data import Data
from rex_xai.lib import ReX
from tqdm import tqdm

from src.config import Config
from src.data import stratified_subset
from src.mnist_data import tensor_to_image
from src.utils.naming import results_location

cfg = Config()

# ReX draws a progress bar for every explanation it extracts, ignoring
# args.progress_bar. With hundreds of images that floods the log, so mute it.
_rex_explanation.tqdm = functools.partial(tqdm, disable=True)

# Kept for sample.py, which imports it.
def tensor_to_pil(x, mean=None, std=None):
    return tensor_to_image(x)


def record(data, path="results/ood.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(data)


def make_rex(model, device):
    """Build a ReX explainer configured for the current dataset's image size."""
    shape = ["N", *cfg.image_shape]
    # "RGB" is ReX's image mode. It reads the channel count from the shape, so
    # a [N, 1, 28, 28] shape works for greyscale MNIST too.
    rex = ReX(model, shape, device, "RGB")
    rex.args.minimum_confidence_threshold = 0.5
    rex.args.batch_size = cfg.rex_batch_size
    rex.args.seed = cfg.seed
    rex.args.min_box_size = cfg.rex_min_box_size
    rex.args.chunk_size = cfg.rex_min_box_size
    rex.args.iters = cfg.rex_iters
    rex.args.insertion_step = cfg.rex_insertion_step
    rex.args.progress_bar = False
    rex.args.verbosity = 0
    return rex, shape


def explain(rex, shape, x, device):
    """Run ReX on one image x ([C, H, W]). Returns the filled-in rex object."""
    x = x.to(device)
    d = Data(
        input=tensor_to_image(x),
        model_shape=shape,
        device=device,
        mode="RGB",
        process=False,  # we hand ReX the already-transformed tensor below
    )
    d.data = x.unsqueeze(0)
    rex.data = d

    rex.set_target()
    rex.calculate_responsibility()
    rex.generate_explanation_object()
    rex.extract_sufficient_explanation()
    return rex


def explanation_size(rex):
    """Share of pixels in the sufficient explanation, in [0, 1]."""
    mask = rex.explanation.sufficiency_mask
    mask = torch.as_tensor(mask).squeeze()
    if mask.ndim == 3:  # [C, H, W]: a pixel counts if any channel is kept
        mask = mask.bool().any(dim=0)
    return torch.count_nonzero(mask).item() / mask.numel()


def measure(model, data, n_samples=cfg.test_size, idx=""):
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    loader = stratified_subset(data, n_samples, seed=cfg.seed)
    rex, shape = make_rex(model, device)
    out_dir = results_location()

    results = []
    for batch_x, batch_y in loader:
        for x, y in zip(batch_x, batch_y):
            with torch.no_grad():
                probs = torch.softmax(model(x.unsqueeze(0).to(device)).float(), dim=1)[0].cpu()
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()

            explain(rex, shape, x, device)

            true_class = int(y)
            pred_class = int(rex.data.target.classification)
            size = explanation_size(rex)

            row = {
                "idx": idx,
                "true_class": true_class,
                "pred_class": pred_class,
                "correct": int(pred_class == true_class),
                "size": size,
                "confidence": probs[pred_class].item(),
                "entropy": entropy,
                # entropy as a share of its maximum, log(num_classes)
                "norm_entropy": entropy / math.log(len(probs)),
                "p_junk": probs[cfg.ood_label].item(),
            }
            results.append(row)
            record(row, f"{out_dir}/per_image.csv")

            if len(results) % 25 == 0:
                sizes = [r["size"] for r in results]
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"  [{idx}] {len(results)}/{n_samples}  mean size={statistics.mean(sizes):.3f}  acc={acc:.3f}")

    groups = {"overall": results}
    for cls in sorted({r["true_class"] for r in results}):
        groups[cls] = [r for r in results if r["true_class"] == cls]

    for cls, rows in groups.items():
        sizes = [r["size"] for r in rows]
        correct = sum(r["correct"] for r in rows)
        record(
            {
                "idx": idx,
                "class": cls,
                "mean": statistics.mean(sizes),
                "stdev": statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
                "accuracy": correct / len(rows),
                "correct": correct,
                "total": len(rows),
            },
            f"{out_dir}/per_class.csv",
        )

    sizes = [r["size"] for r in results]
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"checkpoint {idx}: mean explanation size={statistics.mean(sizes):.3f}  accuracy={acc:.3f}")
    return results
