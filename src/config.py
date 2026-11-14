import os
from dataclasses import dataclass

# SMOKE=1 shrinks everything (data, model, epochs, ReX search) so the whole
# pipeline can be checked end to end on a laptop CPU in a few minutes.
SMOKE = os.environ.get("SMOKE", "0") == "1"

# The images are normalised with ImageNet statistics everywhere (training,
# evaluation and ReX). Keep one copy so un-normalising for plots matches.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Config:
    # data
    data_root: str = "./data"
    imagenet_root: str = "./imagenet"
    # The ~15,000 ImageNet "junk" images the mix actually uses, saved once by
    # `python main.py prepare`. When present, ImageNet itself is not needed.
    junk_cache: str = "./data/junk_cache"
    num_workers: int = 0 if SMOKE else 8
    # 128 for the 16 GB RTX 5080 (also used on the 24 GB L4). ViT-B/16 in bf16 with
    # fused attention should fit; drop to 96 or 64 if the first step runs out of memory.
    batch_size: int = 8 if SMOKE else 128
    eval_batch_size: int = 16 if SMOKE else 256
    # Caps CIFAR train/test sizes in smoke mode only.
    smoke_limit: int | None = 64 if SMOKE else None

    # training
    # Stav's current setup (upstream "Refinements", 2026-10-21): 12 epochs of
    # CIFAR pretraining, then 60 epochs of junk fine-tuning.
    pretrain_epochs: int = 2 if SMOKE else 12
    epochs: int = 2 if SMOKE else 60  # junk fine-tuning epochs = measured checkpoints 1..N
    save_every: int = 1
    # 3e-4 was tuned for batch 64; scaled by sqrt(128/64) for batch 128
    # (square-root scaling, the usual rule for Adam when no warmup is added).
    lr: float = 3e-4 * (128 / 64) ** 0.5  # ~4.24e-4
    wd: float = 0.05
    label_smoothing: float = 0.1

    # model
    # ViT-Base/16 trained from scratch, as upstream since "Usability fixes"
    # (2026-11-05); earlier runs used the MLP-Mixer "mixer_b16_224".
    model: str = "vit_tiny_patch16_224" if SMOKE else "vit_base_patch16_224"
    num_classes: int = 101
    ood_label: int = 100

    # ReX
    rex_batch_size: int = 16 if SMOKE else 128
    rex_iters: int = 1 if SMOKE else 20

    # misc
    seed: int = 42
