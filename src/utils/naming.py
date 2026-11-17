import os

from src.config import Config
cfg = Config()


def pretrained_model_location(model_name=cfg.model):
    return f"models/{model_name}/pretrained.pth"


def _job_id():
    return os.environ.get("JOB_ID", "local")


def model_location(checkpoint_number, model_name=cfg.model):
    """Fine-tuned (junk or no-junk) weights, one folder per job id, so a new
    run never overwrites another run's checkpoints. pretrained.pth (checkpoint
    0) stays shared: every run fine-tunes from the same starting model."""
    return f"models/{model_name}/{_job_id()}/epoch_{checkpoint_number}.pth"


def legacy_model_location(checkpoint_number, model_name=cfg.model):
    """Where runs before per-job folders (e.g. vit1) saved epoch_N.pth."""
    return f"models/{model_name}/epoch_{checkpoint_number}.pth"


def pretrain_epoch_location(epoch, model_name=cfg.model):
    return f"models/{model_name}/pretrain_epoch_{epoch}.pth"


def resume_location(phase, model_name=cfg.model):
    """Full training state (weights, optimiser, schedule) for `--resume`.
    The fine-tuning state is per job id, like its checkpoints."""
    if phase == "train":
        return f"models/{model_name}/{_job_id()}/train_resume.pt"
    return f"models/{model_name}/{phase}_resume.pt"


def results_location(model_name=cfg.model):
    return f"results/{model_name}/{_job_id()}"