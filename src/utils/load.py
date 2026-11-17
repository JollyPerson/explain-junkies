import timm
import torch

import os

from src.utils.naming import pretrained_model_location, model_location, legacy_model_location
from src.utils.device import get_device
from src.config import Config
cfg = Config()


def create_model(model=cfg.model):
    # pretrained=False: every caller overwrites the weights with a checkpoint,
    # so downloading timm's ImageNet weights (~240 MB) first is wasted time and
    # fails on machines without internet access.
    return timm.create_model(model, pretrained=False, num_classes=cfg.num_classes)


def checkpoint_location(epoch):
    if epoch == 0:
        return pretrained_model_location()
    path = model_location(epoch)
    legacy = legacy_model_location(epoch)
    # Runs saved before per-job folders keep working: JOB_ID=vit1 finds
    # models/<model>/epoch_N.pth when models/<model>/vit1/ doesn't have it.
    if not os.path.exists(path) and os.path.exists(legacy):
        return legacy
    return path


def load_weights(model, epoch):
    """Load checkpoint `epoch` (0 = pretrained) into an existing model in place."""
    location = checkpoint_location(epoch)
    print(f"Loading {cfg.model} at checkpoint {epoch} from {location}")
    state = torch.load(location, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def load_model(model=cfg.model, epoch=0):
    m = create_model(model)
    load_weights(m, epoch)
    m.to(get_device())
    return m
