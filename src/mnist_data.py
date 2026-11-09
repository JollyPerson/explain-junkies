"""MNIST (in-distribution) and junk (out-of-distribution) data.

The original experiment uses CIFAR-100 as the real classes and ImageNet as the
junk class. Here the real classes are the 10 MNIST digits (labels 0-9) and every
junk image gets label 10 (cfg.ood_label).

Junk sources (pick with the OOD environment variable):
  fashion    FashionMNIST: same size, same greyscale format, but clothes, not digits.
             Because the format matches, the model can't separate junk by
             resolution or colour alone (unlike CIFAR vs ImageNet).
  synthetic  Random pen strokes and blobs drawn on the fly. Closer to digits in
             style (white strokes on black), so it is a harder junk class.
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.config import Config

cfg = Config()

MNIST_LABELS = {i: str(i) for i in range(10)} | {10: "junk"}


def _transforms(train):
    tf = []
    if train:
        # small shifts/rotations; no flips, since a flipped digit is a different shape
        tf.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    tf += [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
    return transforms.Compose(tf)


def get_mnist(root=None):
    root = root or os.path.join(cfg.data_root, "mnist")
    train = datasets.MNIST(root, train=True, download=True, transform=_transforms(True))
    test = datasets.MNIST(root, train=False, download=True, transform=_transforms(False))
    return train, test


class Relabel(Dataset):
    """Wrap a dataset so every item gets the same label (the junk label)."""

    def __init__(self, base, label):
        self.base, self.label = base, label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.label


class SyntheticJunk(Dataset):
    """Deterministic random scribbles: index i always gives the same image."""

    def __init__(self, n, transform, seed=0, size=28):
        self.n, self.transform, self.seed, self.size = n, transform, seed, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        img = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.6:  # a pen stroke
                pts = [(rng.randint(3, 24), rng.randint(3, 24)) for _ in range(rng.randint(2, 5))]
                draw.line(pts, fill=255, width=rng.randint(1, 3))
            else:  # a blob
                x, y = rng.randint(2, 20), rng.randint(2, 20)
                w, h = rng.randint(3, 8), rng.randint(3, 8)
                draw.ellipse([x, y, x + w, y + h], fill=rng.randint(128, 255))
        return self.transform(img), cfg.ood_label


def get_mnist_ood(source=None, root=None):
    source = source or cfg.ood_source
    root = root or cfg.data_root

    if source == "fashion":
        fr = os.path.join(root, "fashion_mnist")
        train = datasets.FashionMNIST(fr, train=True, download=True, transform=_transforms(True))
        test = datasets.FashionMNIST(fr, train=False, download=True, transform=_transforms(False))
        return Relabel(train, cfg.ood_label), Relabel(test, cfg.ood_label)

    if source == "synthetic":
        return (
            SyntheticJunk(60_000, _transforms(True), seed=1),
            SyntheticJunk(10_000, _transforms(False), seed=2),
        )

    raise ValueError(f"Unknown OOD source {source!r}; use 'fashion' or 'synthetic'")


def tensor_to_image(x):
    """[C, H, W] normalised tensor -> PIL image (greyscale or RGB)."""
    x = x.detach().cpu().float()
    mean = torch.tensor(cfg.mean).view(-1, 1, 1)
    std = torch.tensor(cfg.std).view(-1, 1, 1)
    x = (x * std + mean).clamp(0, 1)
    arr = (x * 255).byte().permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr)
