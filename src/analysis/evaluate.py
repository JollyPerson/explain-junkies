import torch
import statistics
from torch.utils.data import DataLoader
import os
import csv

from rex_xai.lib import ReX
from rex_xai.input.input_data import Data

from src.utils.naming import results_location
from src.config import Config
cfg = Config()

from PIL import Image

job_id = os.environ.get("JOB_ID", "local")
assert job_id, "No job ID"


def tensor_to_pil(x, mean, std):
    # x: [3, H, W]
    x = x.detach().cpu().clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def record(data, path="results/ood.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(data)


def measure(model, data, idx=""):
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    rex = ReX(model, ["N", 3, 224, 224], device, "RGB")
    rex.args.minimum_confidence_threshold = 0.5
    rex.args.batch_size = 128
    rex.args.seed = 0

    results = []

    correct = 0
    total = 0

    for x, y in loader:
        x = x.squeeze(0).to(device, non_blocking=True)
        y = y.to(device)

        if x.ndim == 3:
            x_batch = x.unsqueeze(0)
        else:
            x_batch = x

        d = Data(
            input=tensor_to_pil(x, mean=mean, std=std),
            model_shape=["N", 3, 224, 224],
            device=device,
            mode="RGB",
            process=False,
        )

        d.data = x_batch
        rex.data = d

        rex.set_target()

        pred_class = int(rex.data.target.classification)
        true_class = int(y.item())
        is_correct = pred_class == true_class

        correct += int(is_correct)
        total += 1

        rex.calculate_responsibility()
        rex.generate_explanation_object()
        rex.extract_sufficient_explanation()

        mask = rex.explanation.sufficiency_mask
        size = torch.count_nonzero(mask).item() / mask.numel()

        results.append(
            (size, true_class, pred_class, is_correct)
        )

        if total % 10 == 0:
            sizes = [r[0] for r in results]

            print("mean:", sum(sizes) / len(sizes))
            print(
                "stdev:",
                statistics.stdev(sizes)
                if len(sizes) > 1
                else 0.0,
            )
            print("accuracy:", correct / total)

    sizes = [r[0] for r in results]

    
    # Results shape: (size, true_class, pred_class, is_correct)
    for cls in sorted(set(r[1] for r in results)):
        cls_rows = [r for r in results if r[1] == cls]

        cls_sizes = [r[0] for r in cls_rows]
        cls_correct = sum(1 for r in cls_rows if r[3])
        cls_total = len(cls_rows)

        record(
            {
                "idx": idx,
                "class": cls,
                "mean": sum(cls_sizes) / len(cls_sizes),
                "stdev": (
                    statistics.stdev(cls_sizes)
                    if len(cls_sizes) > 1
                    else 0.0
                ),
                "accuracy": cls_correct / cls_total,
                "correct": cls_correct,
                "total": cls_total,
            },
            f"{results_location(cfg.model)}/per_class.csv",
        )