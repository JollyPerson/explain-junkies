import torch

from src.models import build_model
from src.utils.naming import pretrained_model_location, model_location
from src.config import Config
cfg = Config()


def load_model(model=cfg.model, epoch=0):
    """Load a checkpoint. epoch=0 is the pretrained model (no junk class seen yet)."""
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    print(f"Loading {model} at checkpoint {epoch}")
    location = pretrained_model_location() if epoch == 0 else model_location(epoch)

    # pretrained=False: the weights come from our own checkpoint, so there is
    # no need to download ImageNet weights only to overwrite them.
    net = build_model(model, num_classes=cfg.num_classes, in_channels=cfg.image_shape[0])
    state = torch.load(location, map_location="cpu", weights_only=False)

    net.load_state_dict(state)
    net.to(device)

    return net
