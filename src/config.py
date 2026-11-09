import os
from dataclasses import dataclass, field

# Which experiment to run. Pick it with an environment variable so every module
# (data, training, ReX measurement, plots) sees the same choice:
#
#   DATASET=mnist  python main.py all     # default: small CNN, runs on a laptop CPU
#   DATASET=cifar  python main.py all     # original CIFAR-100 + ImageNet setup (needs a GPU)
#
# For MNIST, OOD picks the "junk" data that gets the extra label:
#   OOD=fashion    FashionMNIST clothing images (default, auto-downloaded)
#   OOD=synthetic  random strokes and blobs generated on the fly (no download)
DATASET = os.environ.get("DATASET", "mnist").lower()
OOD = os.environ.get("OOD", "fashion").lower()

if DATASET not in ("mnist", "cifar"):
    raise ValueError(f"DATASET must be 'mnist' or 'cifar', got {DATASET!r}")


def _pick(mnist, cifar):
    """Choose a default depending on the dataset, evaluated once at import."""
    return field(default=mnist if DATASET == "mnist" else cifar)


@dataclass(frozen=True)
class Config:
    # data
    dataset: str = DATASET
    ood_source: str = _pick(OOD, "imagenet")
    data_root: str = "./data"
    imagenet_root: str = "./imagenet"
    num_workers: int = _pick(2, 8)
    batch_size: int = _pick(128, 64)
    ood_fraction: float = 0.2  # share of the mixed dataset that is junk

    # The tensor shape the model sees, without the batch dimension (C, H, W).
    # ReX needs this to know how to occlude the input.
    image_shape: tuple = _pick((1, 28, 28), (3, 224, 224))
    # Normalisation applied by the data transforms, also used to turn tensors
    # back into viewable images. MNIST is left in [0, 1] on purpose: ReX masks
    # with the value 0, and 0 is then the black background of a digit image.
    mean: tuple = _pick((0.0,), (0.485, 0.456, 0.406))
    std: tuple = _pick((1.0,), (0.229, 0.224, 0.225))

    # training
    pretrain_epochs: int = _pick(3, 12)
    epochs: int = _pick(10, 60)  # epochs of fine-tuning on the mixed (junk) dataset
    save_every: int = 1
    lr: float = _pick(1e-3, 3e-4)
    wd: float = _pick(1e-4, 0.05)

    # model
    model: str = _pick("small_cnn", "vit_base_patch16_224")
    num_classes: int = _pick(11, 101)  # real classes + 1 junk class
    ood_label: int = _pick(10, 100)    # the junk class is always the last one

    # misc
    seed: int = 42
    test_size: int = _pick(200, 100)  # images ReX explains per checkpoint

    # ReX search settings. The defaults in rex-xai are tuned for 224px images;
    # on 28x28 a 10px minimum box would be a third of the image.
    rex_min_box_size: int = _pick(2, 10)
    rex_iters: int = _pick(10, 20)
    rex_batch_size: int = _pick(256, 128)
    rex_insertion_step: int = _pick(8, 100)
