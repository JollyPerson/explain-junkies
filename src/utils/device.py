import os
import random
import contextlib

import numpy as np
import torch


def get_device():
    """CUDA if present, then Apple's MPS (M-series Macs), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(_worker_id):
    # torch already gives each DataLoader worker its own seed derived from the
    # loader's generator; pass it on to numpy and random as well.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed):
    return torch.Generator().manual_seed(seed)


def amp_dtype(device, precision="auto"):
    """Pick the autocast dtype, or None for full fp32.

    "auto": bf16 on GPUs that support it (Ampere and newer), fp16 on older CUDA
    GPUs, fp32 elsewhere. On Pascal cards (GTX 10xx) fp16 saves memory but
    gives little speed, since they have no tensor cores.
    """
    if precision == "fp32" or device != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)
