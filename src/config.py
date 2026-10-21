from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # data
    data_root: str = "./data"
    imagenet_root: str = "./imagenet"
    num_workers: int = 8
    batch_size: int = 64

    # training
    epochs: int = 60
    save_every: int = 1
    lr: float = 3e-4
    wd: float = 0.05

    # model
    model: str = "mixer_b16_224"
    num_classes: int = 101
    ood_label: int = 100

    # misc
    seed: int = 42
    test_size: int = 100
