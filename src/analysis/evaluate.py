import csv
import os
import math
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image

import functools

import rex_xai.explanation.explanation as _rex_explanation
from rex_xai.lib import ReX
from tqdm import tqdm
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction

from src.data import targets_of
from src.utils.naming import results_location
from src.utils.device import get_device, amp_dtype, autocast
from src.config import Config, MEAN, STD
cfg = Config()

# ReX draws a progress bar for every image it explains, which floods the log
# with thousands of lines. Turn those bars off.
_rex_explanation.tqdm = functools.partial(tqdm, disable=True)


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


def _sizes(rows):
    """Explanation sizes, leaving out images ReX found no explanation for (NaN)."""
    return [r["size"] for r in rows if not math.isnan(r["size"])]


def _make_rex(model, device, dtype):
    rex = ReX(model, ["N", 3, 224, 224], device, "RGB",
              prediction_function=make_prediction_function(model, device, dtype))
    rex.args.minimum_confidence_threshold = 0.5
    rex.args.batch_size = cfg.rex_batch_size
    rex.args.iters = cfg.rex_iters
    # ReX reseeds from this at the start of every image, so each image's
    # random splits are the same whatever order (or process) explains it.
    rex.args.seed = 0
    rex.args.progress_bar = False
    return rex


def _explain_one(rex, x, y, idx, image):
    """Run ReX on one image ([1, 3, 224, 224] on the device) and return its row."""
    d = Data(
        input=tensor_to_pil(x.squeeze(0)),
        model_shape=["N", 3, 224, 224],
        device=x.device.type,
        mode="RGB",
        process=False,
    )
    d.data = x
    rex.data = d

    rex.set_target()

    pred_class = int(rex.data.target.classification)
    true_class = int(y)

    rex.calculate_responsibility()
    if rex.maps is None or rex.maps.get(rex.data.target.classification) is None:
        # No occluded version of the image kept the predicted class, so ReX
        # has no responsibility map to build an explanation from. Happens
        # with low-confidence predictions (e.g. the untrained SMOKE model).
        # Record it, but leave it out of the size averages.
        size = float("nan")
    else:
        rex.generate_explanation_object()
        rex.extract_sufficient_explanation()
        mask = rex.explanation.sufficiency_mask
        size = torch.count_nonzero(mask).item() / mask.numel()

    return {
        "idx": idx,
        "image": image,
        "true_class": true_class,
        "pred_class": pred_class,
        "correct": int(pred_class == true_class),
        "confidence": float(rex.data.target.confidence),
        "size": size,
    }


def _progress(rows, done, total, start, prefix=""):
    sizes = _sizes(rows)
    mean = statistics.fmean(sizes) if sizes else float("nan")
    correct = sum(r["correct"] for r in rows)
    no_explanation = len(rows) - len(sizes)
    per_img = (time.time() - start) / done
    eta_min = per_img * (total - done) / 60
    print(f"{prefix}[{done}/{total}] mean={mean:.4f} "
          f"accuracy={correct / len(rows):.4f}  no_explanation={no_explanation}  "
          f"{per_img:.2f}s/image  checkpoint ETA {eta_min:.0f} min", flush=True)


def _write_results(idx, rows):
    """Write the per-image CSV and the per-class summary for checkpoint `idx`."""
    rows = sorted(rows, key=lambda r: r["image"])
    _write_csv(per_image_path(idx), rows)

    summary = []
    for cls in sorted({r["true_class"] for r in rows}):
        cls_rows = [r for r in rows if r["true_class"] == cls]
        cls_sizes = _sizes(cls_rows)
        cls_correct = sum(r["correct"] for r in cls_rows)
        summary.append({
            "idx": idx,
            "class": cls,
            "mean": statistics.fmean(cls_sizes) if cls_sizes else float("nan"),
            "stdev": statistics.stdev(cls_sizes) if len(cls_sizes) > 1 else 0.0,
            "accuracy": cls_correct / len(cls_rows),
            "correct": cls_correct,
            "total": len(cls_rows),
            "no_explanation": len(cls_rows) - len(cls_sizes),
        })
    _write_csv(per_class_path(idx), summary)


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
    rex = _make_rex(model, device, dtype)

    rows = []
    start = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)  # [1, 3, 224, 224]
        rows.append(_explain_one(rex, x, y.item(), idx, i))
        if (i + 1) % 50 == 0:
            _progress(rows, i + 1, len(data), start)

    print(f"checkpoint {idx}: {len(rows)} images explained in {(time.time() - start) / 60:.1f} min")
    _write_results(idx, rows)
    return rows


# ---------------------------------------------------------------------------
# Parallel measure: several processes share one GPU.
#
# ReX spends most of its time in single-threaded Python between small GPU
# calls, so one process leaves the GPU idle most of the time. Each worker
# loads its own copy of the model and explains every Nth image (image i goes
# to worker i % workers). Results are identical to the single-process version
# because ReX reseeds for every image; only the order of work changes.
# ---------------------------------------------------------------------------

_worker_data = None


def _init_worker(data):
    global _worker_data
    _worker_data = data
    # Every worker has its own Python thread; extra torch CPU threads per
    # worker would just fight each other for the same cores.
    torch.set_num_threads(1)


def _worker(job):
    from src.utils.load import create_model, load_weights

    epoch, rank, workers, precision = job
    device = get_device()
    dtype = amp_dtype(device, precision)
    model = create_model()
    load_weights(model, epoch)
    model.eval().to(device)
    rex = _make_rex(model, device, dtype)

    data = _worker_data
    mine = list(range(rank, len(data), workers))
    rows = []
    start = time.time()
    for n, i in enumerate(mine):
        x, y = data[i]
        x = x.unsqueeze(0).to(device)
        rows.append(_explain_one(rex, x, int(y), str(epoch), i))
        # Worker 0 reports for everyone; the others run at the same speed.
        if rank == 0 and (n + 1) % 25 == 0:
            done = (n + 1) * workers
            _progress(rows, done, len(data), start, prefix=f"[worker 0 of {workers}] ")
    return rows


def measure_parallel(epoch, data, workers, precision="fp32"):
    """Same output as measure(), using `workers` processes on one GPU.

    Builds the models inside the workers, so the caller must not have
    loaded a model onto the GPU itself (CUDA cannot be shared that way).
    """
    import torch.multiprocessing as mp

    if len(data) == 0:
        raise ValueError("Dataset is empty")
    idx = str(epoch)
    print(f"Measuring checkpoint {idx} on {len(data)} images with {workers} worker processes, "
          f"device={get_device()}, rex precision={precision}", flush=True)

    start = time.time()
    # "spawn" starts clean processes; CUDA does not survive "fork".
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(data,)) as pool:
        parts = pool.map(_worker, [(epoch, r, workers, precision) for r in range(workers)])
    rows = [r for part in parts for r in part]

    print(f"checkpoint {idx}: {len(rows)} images explained in {(time.time() - start) / 60:.1f} min",
          flush=True)
    _write_results(idx, rows)
    return rows
