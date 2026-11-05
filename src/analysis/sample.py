import os
import argparse
import random
import torch
from torch.utils.data import Subset

from rex_xai.lib import ReX
from rex_xai.input.input_data import Data

from src.analysis.evaluate import tensor_to_pil
from src.data import get_base, get_ood
from src.utils.load import load_model
from src.utils.device import get_device
from src.config import Config
cfg = Config()


CIFAR100_LABELS = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper",
    84: "table", 85: "tank", 86: "telephone", 87: "television", 88: "tiger",
    89: "tractor", 90: "train", 91: "trout", 92: "tulip", 93: "turtle",
    94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman",
    99: "worm",
}


def class_name(y):
    return CIFAR100_LABELS.get(int(y), str(int(y)))


def stratified_indices(dataset, per_class=100, seed=0):
    rng = random.Random(seed)
    by_class = {}

    for i in range(len(dataset)):
        _, y = dataset[i]
        y = int(y)
        by_class.setdefault(y, []).append(i)

    selected = []

    for y, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        selected.extend(indices[:per_class])

    return selected

def save_rex_example(model, x, y, idx, out_dir, device):
    import matplotlib.pyplot as plt

    label = class_name(y)

    heatmap_dir = os.path.join(out_dir, "heatmaps", f"{int(y):03d}_{label}")
    surface_dir = os.path.join(out_dir, "surface", f"{int(y):03d}_{label}")
    mask_dir = os.path.join(out_dir, "sufficiency_mask", f"{int(y):03d}_{label}")
    input_dir = os.path.join(out_dir, "inputs", f"{int(y):03d}_{label}")

    for dpath in [heatmap_dir, surface_dir, mask_dir, input_dir]:
        os.makedirs(dpath, exist_ok=True)

    x = x.to(device)
    x_batch = x.unsqueeze(0) if x.ndim == 3 else x

    rex = ReX(model, ["N", 3, 224, 224], device, "RGB")
    rex.args.minimum_confidence_threshold = 0.5
    rex.args.batch_size = cfg.rex_batch_size
    rex.args.iters = cfg.rex_iters
    rex.args.seed = 0

    d = Data(
        input=tensor_to_pil(x),
        model_shape=["N", 3, 224, 224],
        device=device,
        mode="RGB",
        process=False,
    )

    d.data = x_batch
    rex.data = d

    rex.set_target()
    rex.calculate_responsibility()
    rex.generate_explanation_object()
    rex.extract_sufficient_explanation()

    pred = int(rex.data.target.classification)
    pred_label = class_name(pred)

    fname = f"idx_{idx}_true_{int(y)}_{label}_pred_{pred}_{pred_label}"

    d.input.save(os.path.join(input_dir, fname + ".png"))

    rex.explanation.heatmap_plot(
        path=os.path.join(heatmap_dir, fname + ".png")
    )

    rex.explanation.surface_plot(
        path=os.path.join(surface_dir, fname + ".png")
    )

    mask = rex.explanation.sufficiency_mask

    if torch.is_tensor(mask):
        mask = mask.detach().cpu()

    mask = mask.squeeze()

    if mask.ndim == 3 and mask.shape[0] in (1, 3):
        mask = mask.any(dim=0) if torch.is_tensor(mask) else mask.any(axis=0)

    if torch.is_tensor(mask):
        mask = mask.numpy()

    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(mask_dir, fname + ".png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    print("saved", fname)

def sample_rex_examples(model, dataset, out_dir="visualizations/examples", per_class=100, seed=0):
    device = get_device()
    model.eval().to(device)

    indices = stratified_indices(dataset, per_class=per_class, seed=seed)

    for idx in indices:
        x, y = dataset[idx]
        save_rex_example(model, x, y, idx, out_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="visualizations/examples")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=cfg.epochs,
                        help="Checkpoint to explain (0 = pretrained)")
    args = parser.parse_args()
    
    print("loading model")
    # models/ood_59.pth was the checkpoint name before the refactor.
    model = load_model(epoch=args.epoch)

    print("getting data")
    data, _ = get_ood()

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(data), generator=generator)[:500].tolist()
    test = Subset(data, indices)

    print("getting examples")
    sample_rex_examples(
        model,
        test,
        out_dir=args.out_dir,
        per_class=10,
        seed=args.seed,
    )
