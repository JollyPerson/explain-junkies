import os

from src.config import Config
cfg = Config()


def pretrained_model_location(model_name=cfg.model):
    return f"models/{model_name}/pretrained.pth"


def model_location(checkpoint_number, model_name=cfg.model):
    return f"models/{model_name}/epoch_{checkpoint_number}.pth"


def pretrain_epoch_location(epoch, model_name=cfg.model):
    return f"models/{model_name}/pretrain_epoch_{epoch}.pth"


def resume_location(phase, model_name=cfg.model):
    """Full training state (weights, optimiser, schedule) for `--resume`."""
    return f"models/{model_name}/{phase}_resume.pt"


def results_location(model_name=cfg.model):
    job_id = os.environ.get("JOB_ID", "local")
    return f"results/{model_name}/{job_id}"