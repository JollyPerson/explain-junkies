import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from collections import defaultdict

from tqdm import tqdm

from src.config import Config, MEAN, STD
cfg = Config()

IN_CIFAR_CLASSES = [
    0, 1, 2, 3, 4, 5, 6, 107, 118, 119, 120, 121, 122, 124, 147, 148, 150,
    25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    300, 301, 302, 303, 304, 305, 306, 307,
    308, 309, 310, 311, 312, 314, 315, 319, 320,
    321, 322, 323, 324, 325, 326,
    333, 334, 337, 360, 361,
    269, 270, 271, 272, 273, 274, 275,
    277, 278, 279, 280,
    286, 288, 291, 292, 293,
    365, 366, 367,
    385, 386,
    80, 81, 82, 83, 84, 85, 86,
    127, 128, 129, 130,
    144, 145, 146,
    429, 430, 444, 487, 504, 610, 620, 621, 673,
    717, 779, 805, 829, 846, 847, 851, 866, 923, 968
]


class FilterByClassIdx(Dataset):
    """ImageNet minus the classes that overlap CIFAR-100, relabelled as junk."""

    def __init__(self, base_ds, excluded_idxs, k=10_000):
        self.base_ds = base_ds
        self.excluded = set(excluded_idxs)
        self.ood_label = cfg.ood_label

        # k (10,000) is above ImageNet's ~1,300 images per class, so in practice
        # this keeps every image of every non-excluded class.
        counts = defaultdict(int)
        self.keep_indices = []
        for i, y in enumerate(base_ds.targets):
            if y in self.excluded or counts[y] >= k:
                continue
            counts[y] += 1
            self.keep_indices.append(i)

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        x, _ = self.base_ds[self.keep_indices[idx]]
        return x, self.ood_label


def _get_transforms(mean=MEAN, std=STD):
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf


def targets_of(ds):
    """Labels of a dataset without loading any images."""
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if isinstance(ds, Subset):
        parent = targets_of(ds.dataset)
        return [parent[i] for i in ds.indices]
    if isinstance(ds, ConcatDataset):
        return [y for d in ds.datasets for y in targets_of(d)]
    if isinstance(ds, (FilterByClassIdx, JunkCache)):
        return [cfg.ood_label] * len(ds)
    raise TypeError(f"Can't read labels from {type(ds).__name__}")


# ---------------------------------------------------------------- CIFAR-100

def get_base(data_root="./data/cifar"):
    print("Loading CIFAR")
    train_tf, test_tf = _get_transforms()

    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)

    if cfg.smoke_limit:
        train_ds = Subset(train_ds, range(cfg.smoke_limit))
        test_ds = Subset(test_ds, range(cfg.smoke_limit))

    return train_ds, test_ds


# ------------------------------------------------------------ ImageNet junk

def _imagenet_split(imagenet_root, split, transform, use_parquet):
    if use_parquet:
        from src.parquet_data import ParquetImageNet
        return ParquetImageNet(imagenet_root, "validation" if split == "val" else split, transform=transform)
    return datasets.ImageNet(root=imagenet_root, split=split, transform=transform)


def get_ood(imagenet_root="./data/imagenet", excluded_idxs=IN_CIFAR_CLASSES,
            use_parquet=False, train_tf=None, test_tf=None):
    print("Getting ImageNet")
    default_train, default_test = _get_transforms()
    train_base = _imagenet_split(imagenet_root, "train", train_tf or default_train, use_parquet)
    val_base = _imagenet_split(imagenet_root, "val", test_tf or default_test, use_parquet)
    return FilterByClassIdx(train_base, excluded_idxs), FilterByClassIdx(val_base, excluded_idxs)


def select_mix_indices(len_a, len_b, p=0.2, total_size=None, seed=0):
    """Which items of a (CIFAR) and b (junk) go into the mix.

    Same arithmetic and random draws as the original mix_datasets, so the
    selection is unchanged. With CIFAR's 50,000 train / 10,000 test images and
    p=0.2 this picks 12,500 junk train images and 2,500 junk test images.
    """
    assert 0.0 <= p <= 1.0

    if total_size is None:
        if p == 0:
            total_size = len_a
        elif p == 1:
            total_size = len_b
        else:
            total_size = min(int(len_a / (1 - p)), int(len_b / p))

    n_b = int(round(total_size * p))
    n_a = total_size - n_b

    if n_a > len_a or n_b > len_b:
        raise ValueError(f"Requested n_a={n_a}, n_b={n_b}, but len(a)={len_a}, len(b)={len_b}")

    g = torch.Generator().manual_seed(seed)
    idx_a = torch.randperm(len_a, generator=g)[:n_a].tolist()
    idx_b = torch.randperm(len_b, generator=g)[:n_b].tolist()
    return idx_a, idx_b


def mix_datasets(a, b, p=0.2, total_size=None, seed=0):
    idx_a, idx_b = select_mix_indices(len(a), len(b), p, total_size, seed)
    return ConcatDataset([Subset(a, idx_a), Subset(b, idx_b)])


class JunkCache(Dataset):
    """The selected junk images, pre-resized to 224x224 uint8 in one .npy file.

    The file is memory-mapped, so DataLoader workers share it instead of each
    holding a copy, and nothing is decoded from JPEG or parquet at train time.
    """

    def __init__(self, root, split, transform=None):
        self.path = os.path.join(root, f"{split}_images.npy")
        with open(os.path.join(root, f"{split}_meta.json")) as f:
            self.meta = json.load(f)
        self.transform = transform
        self._arr = None

    def __len__(self):
        return self.meta["n"]

    def __getitem__(self, idx):
        if self._arr is None:  # opened lazily, once per worker
            self._arr = np.load(self.path, mmap_mode="r")
        x = Image.fromarray(np.array(self._arr[idx]))
        if self.transform is not None:
            x = self.transform(x)
        return x, cfg.ood_label


def junk_cache_exists(root=cfg.junk_cache):
    return all(
        os.path.exists(os.path.join(root, f"{s}_{kind}"))
        for s in ("train", "test") for kind in ("images.npy", "meta.json")
    )


class _Uint8(Dataset):
    def __init__(self, ds, indices):
        self.ds, self.indices = ds, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, _ = self.ds[self.indices[i]]
        return torch.from_numpy(np.asarray(x, dtype=np.uint8).copy())


def _default_imagenet_root(imagenet_root, use_parquet):
    if imagenet_root:
        return imagenet_root
    return cfg.imagenet_root if use_parquet else "./data/imagenet"


def build_junk_cache(imagenet_root=None, use_parquet=False, out_dir=cfg.junk_cache):
    """Save the junk images the mix uses, so later runs never touch ImageNet.

    The images are stored after Resize(224) + CenterCrop(224). Training then
    applies RandomCrop(224, padding=8) and a flip to that square, so the only
    change from reading ImageNet directly is that the random crop no longer
    slides along the long side of non-square photos.
    """
    imagenet_root = _default_imagenet_root(imagenet_root, use_parquet)
    os.makedirs(out_dir, exist_ok=True)
    base_train, base_test = get_base()
    to_square = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    ood_train, ood_test = get_ood(imagenet_root, use_parquet=use_parquet,
                                  train_tf=to_square, test_tf=to_square)

    for split, base, ood in (("train", base_train, ood_train), ("test", base_test, ood_test)):
        _, idx_b = select_mix_indices(len(base), len(ood))
        # Read in file order (much faster for parquet row groups and disk
        # caches), but store in the mix's order.
        order = sorted(range(len(idx_b)), key=lambda i: idx_b[i])
        loader = DataLoader(_Uint8(ood, [idx_b[i] for i in order]), batch_size=64,
                            shuffle=False, num_workers=cfg.num_workers)

        path = os.path.join(out_dir, f"{split}_images.npy")
        tmp = path + ".partial.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(idx_b), 224, 224, 3))
        pos = 0
        for batch in tqdm(loader, desc=f"caching junk {split}"):
            for img in batch.numpy():
                out[order[pos]] = img
                pos += 1
        out.flush()
        del out
        os.replace(tmp, path)

        source = [ood.keep_indices[i] for i in idx_b]
        meta = {
            "n": len(idx_b),
            "len_b": len(ood),
            "imagenet_index": source,
            "imagenet_class": [int(ood.base_ds.targets[i]) for i in source],
            "use_parquet": use_parquet,
            "excluded_classes": sorted(ood.excluded),
        }
        with open(os.path.join(out_dir, f"{split}_meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"saved {len(idx_b)} junk {split} images to {path}")


def get_mixed_dataset(use_parquet=False, imagenet_root=None):
    base_train, base_test = get_base()

    if junk_cache_exists():
        print(f"Using junk cache at {cfg.junk_cache}")
        train_tf, test_tf = _get_transforms()
        mixes = []
        for split, base, tf in (("train", base_train, train_tf), ("test", base_test, test_tf)):
            junk = JunkCache(cfg.junk_cache, split, transform=tf)
            idx_a, _ = select_mix_indices(len(base), junk.meta["len_b"])
            mixes.append(ConcatDataset([Subset(base, idx_a), junk]))
        return tuple(mixes)

    print("No junk cache found; reading ImageNet directly (run `python main.py prepare` to speed this up)")
    ood_train, ood_test = get_ood(_default_imagenet_root(imagenet_root, use_parquet), use_parquet=use_parquet)
    return mix_datasets(base_train, ood_train), mix_datasets(base_test, ood_test)
