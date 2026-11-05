import bisect
import glob
import io
import os

import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from src.data import IN_CIFAR_CLASSES, FilterByClassIdx, _get_transforms
from src.config import Config

cfg = Config()


class ParquetImageNet(Dataset):
    """Lazy ImageNet reader for Hugging Face parquet shards."""

    def __init__(self, root, split, transform=None):
        self.files = self._find_files(root, split)
        self.transform = transform
        self._row_groups = []
        self._offsets = [0]

        for path in self.files:
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                rows = parquet.metadata.row_group(row_group).num_rows
                self._row_groups.append((path, row_group))
                self._offsets.append(self._offsets[-1] + rows)

        self._cache_key = None
        self._cache_rows = None
        self.targets = self._read_targets()

    @staticmethod
    def _find_files(root, split):
        split_names = {
            "train": ("train",),
            "validation": ("validation", "val"),
            "test": ("test",),
        }[split]
        files = []
        for name in split_names:
            patterns = (
                os.path.join(root, name, "*.parquet"),
                os.path.join(root, f"{name}*.parquet"),
                os.path.join(root, "data", f"{name}*.parquet"),
                os.path.join(root, "**", f"{name}-*.parquet"),
            )
            files.extend(path for pattern in patterns for path in glob.glob(pattern, recursive=True))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No parquet files found for split {split!r} below {root}")
        return files

    def _read_targets(self):
        targets = []
        dataset = pa_dataset.dataset(self.files, format="parquet")
        for batch in dataset.scanner(columns=["label"], batch_size=65_536).to_batches():
            targets.extend(int(label) for label in batch.column("label").to_pylist())
        if len(targets) != len(self):
            raise ValueError(
                f"Parquet labels contain {len(targets)} rows, expected {len(self)}"
            )
        return targets

    def __len__(self):
        return self._offsets[-1]

    def _load_row_group(self, row_group_index):
        key = self._row_groups[row_group_index]
        if key != self._cache_key:
            self._cache_rows = pq.ParquetFile(key[0]).read_row_group(
                key[1], columns=["image", "label"]
            ).to_pylist()
            self._cache_key = key
        return self._cache_rows

    @staticmethod
    def _decode_image(value):
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            if image_bytes is not None:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = value.get("path")
            if image_path is not None:
                return Image.open(image_path).convert("RGB")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        raise ValueError("Expected image bytes or a path in the parquet image column")

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        row_group_index = bisect.bisect_right(self._offsets, idx) - 1
        row = self._load_row_group(row_group_index)[idx - self._offsets[row_group_index]]
        image = self._decode_image(row["image"])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def get_ood_parquet(
    imagenet_root=None,
    excluded_idxs=IN_CIFAR_CLASSES,
):
    """Return OOD ImageNet datasets from train and validation parquet shards."""
    root = imagenet_root or cfg.imagenet_root
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf, test_tf = _get_transforms(mean, std)

    train_base = ParquetImageNet(root, "train", transform=train_tf)
    val_base = ParquetImageNet(root, "validation", transform=test_tf)
    return (
        FilterByClassIdx(train_base, excluded_idxs),
        FilterByClassIdx(val_base, excluded_idxs),
    )
