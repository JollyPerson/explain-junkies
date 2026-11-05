import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from collections import defaultdict

from tqdm import tqdm

from src.config import Config
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

dataloader_kwargs = {
    "batch_size": cfg.batch_size,
    "shuffle": True,
    "num_workers": cfg.num_workers,
    "pin_memory": True,
    "persistent_workers": (cfg.num_workers > 0),
}



class FilterByClassIdx(Dataset):
    def __init__(self, base_ds, excluded_idxs, k=10_000):
        print("initializing filtered imagenet")
        self.base_ds = base_ds
        self.excluded = set(excluded_idxs)
        self.ood_label = 100

        counts = defaultdict(int)
        self.keep_indices = []

        keep_classes = set(base_ds.targets) - self.excluded

        for i, y in enumerate(base_ds.targets):
            if y in self.excluded:
                continue
            if counts[y] < k:
                counts[y] += 1
                self.keep_indices.append(i)
            
                if all(counts[y] >= k for y in keep_classes):
                    break
        
        print("initialized filtered imagenet")

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        x, _ = self.base_ds[self.keep_indices[idx]]
        return x, self.ood_label



def _get_transforms(mean, std):
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

def _get_mean_std(ds):
    # for testing
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    print("calculating stats")
    # assert torch.cuda.is_available(), "No cuda!"
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0

    for x, _ in tqdm(ds):
        b, c, h, w = x.shape
        pixels = b * h * w
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sq_sum += x.square().sum(dim=(0, 2, 3))
        n_pixels += pixels

    mean = (channel_sum / n_pixels).tolist()
    std = (channel_sq_sum / n_pixels - torch.tensor(mean) ** 2).sqrt().tolist()

    print("calculated stats")
    return mean, std


def get_ood(
    imagenet_root="./data/imagenet",
    excluded_idxs=IN_CIFAR_CLASSES,
):
    print("Getting ImageNet")
    stats_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    try:
        stats_train_base = datasets.ImageNet(root=imagenet_root, split="train", transform=stats_tf)
    except:
        print("Looks like ImageNet is not installed locally! Try with MNIST")
        exit()

    print("filtering imagenet")
    stats_train_ds = DataLoader(
        FilterByClassIdx(stats_train_base, excluded_idxs),
        **dataloader_kwargs
    )

    # mean, std = _get_mean_std(stats_train_ds)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf, test_tf = _get_transforms(mean, std)
    
    train_base = datasets.ImageNet(root=imagenet_root, split="train", transform=train_tf)
    val_base   = datasets.ImageNet(root=imagenet_root, split="val",   transform=test_tf)

    ood_train_ds = FilterByClassIdx(train_base, excluded_idxs)
    ood_val_ds   = FilterByClassIdx(val_base, excluded_idxs)

    ood_train_loader = DataLoader(ood_train_ds, **dataloader_kwargs)
    ood_val_loader = DataLoader(ood_val_ds, **dataloader_kwargs)

    return ood_train_loader, ood_val_loader



def get_base(data_root="./data/cifar"):
    print("Loading CIFAR")
    stats_tf = transforms.Compose([transforms.ToTensor()])

    try:
        stats_train_ds = datasets.CIFAR100(
            root=data_root, train=True, download=False, transform=stats_tf
        )
    except RuntimeError:
        print("Looks like CIFAR is not installed locally! Try with MNIST")
        exit()

    loader = DataLoader(
        stats_train_ds,
        **dataloader_kwargs
    )

    mean, std = _get_mean_std(loader)
    train_tf, test_tf = _get_transforms(mean, std)

    train_ds = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_ds = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_tf,
    )

    train_loader = DataLoader(train_ds, **dataloader_kwargs)
    test_loader = DataLoader(test_ds, **dataloader_kwargs)

    return train_loader, test_loader


def mix_datasets(a: DataLoader, b: DataLoader, p=0.2, total_size=None, seed=0):
    assert 0.0 <= p <= 1.0

    a, b = a.dataset, b.dataset

    if total_size is None:
        if p == 0:
            total_size = len(a)
        elif p == 1:
            total_size = len(b)
        else:
            total_size = min(int(len(a) / (1 - p)), int(len(b) / p))

    n_b = int(round(total_size * p))
    n_a = total_size - n_b

    if n_a > len(a) or n_b > len(b):
        raise ValueError(
            f"Requested n_a={n_a}, n_b={n_b}, but len(a)={len(a)}, len(b)={len(b)}"
        )

    g = torch.Generator().manual_seed(seed)
    idx_a = torch.randperm(len(a), generator=g)[:n_a].tolist()
    idx_b = torch.randperm(len(b), generator=g)[:n_b].tolist()

    mixed = ConcatDataset([
        Subset(a, idx_a),
        Subset(b, idx_b),
    ])

    return DataLoader(mixed, **dataloader_kwargs)

def get_mixed_dataset():
    base_train, base_test = get_base()
    ood_train, ood_test = get_ood()

    mix_train = mix_datasets(base_train, ood_train)
    mix_test = mix_datasets(base_test, ood_test)

    return mix_train, mix_test


def stratified_subset(loader: DataLoader, n=100, seed=0):
    ds = loader.dataset
    n = min(n, len(ds))

    g = torch.Generator().manual_seed(seed)
    edges = torch.linspace(0, len(ds), n + 1).long()

    indices = [
        torch.randint(int(edges[i]), int(edges[i + 1]), (1,), generator=g).item()
        for i in range(n)
    ]

    return DataLoader(
        Subset(ds, indices),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )
