"""Save example ReX explanations as images, to see what the numbers mean.

    python -m src.analysis.sample --epoch 0 --per-class 3     # pretrained model
    python -m src.analysis.sample --epoch 10 --per-class 3    # after junk fine-tuning

For each image it writes the input, ReX's responsibility heatmap, and the
sufficient-explanation mask (the pixels ReX kept), under
visualizations/<run>/examples/epoch_<N>/.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch

from src.analysis.evaluate import explain, explanation_size, make_rex
from src.analysis.visualize import LABELS
from src.config import Config
from src.data import get_mixed_dataset
from src.mnist_data import tensor_to_image
from src.utils.load import load_model
from src.utils.naming import RUN_NAME

cfg = Config()


def class_name(y):
    return LABELS.get(int(y), str(int(y)))


def stratified_indices(dataset, per_class=3, seed=0, max_scan=5000):
    """Pick up to per_class random examples of each label."""
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    picked = {}
    for i in order[:max_scan]:
        y = int(dataset[i][1])
        if len(picked.setdefault(y, [])) < per_class:
            picked[y].append(i)
    return [i for y in sorted(picked) for i in picked[y]]


def save_example(rex, shape, x, y, idx, out_dir, device):
    explain(rex, shape, x, device)

    pred = int(rex.data.target.classification)
    size = explanation_size(rex)
    name = f"idx{idx}_true_{class_name(y)}_pred_{class_name(pred)}_size{size:.2f}"

    mask = torch.as_tensor(rex.explanation.sufficiency_mask).detach().cpu().squeeze()
    if mask.ndim == 3:
        mask = mask.bool().any(dim=0)

    img = tensor_to_image(x)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"input (true {class_name(y)})")
    resp = rex.maps.get(pred)
    if resp is not None:
        axes[1].imshow(resp.squeeze(), cmap="magma")
    axes[1].set_title("ReX responsibility")
    kept = torch.as_tensor(x).squeeze().cpu() * mask
    axes[2].imshow(kept if kept.ndim == 2 else tensor_to_image(kept), cmap="gray")
    axes[2].set_title(f"kept {size:.0%} -> pred {class_name(pred)}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=120)
    plt.close(fig)
    print("saved", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=cfg.epochs, help="checkpoint; 0 = pretrained")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or f"visualizations/{RUN_NAME}/examples/epoch_{args.epoch}"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(epoch=args.epoch).eval()
    rex, shape = make_rex(model, device)

    _, test_loader = get_mixed_dataset()
    test = test_loader.dataset

    for i in stratified_indices(test, per_class=args.per_class, seed=args.seed):
        x, y = test[i]
        save_example(rex, shape, x, y, i, out_dir, device)
