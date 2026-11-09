import os

from src.config import Config
cfg = Config()

# Folder name for this experiment's checkpoints and results, so runs with
# different datasets or junk sources don't overwrite each other.
RUN_NAME = cfg.model if cfg.dataset == "cifar" else f"{cfg.dataset}_{cfg.ood_source}_{cfg.model}"


def pretrained_model_location(model_name=RUN_NAME):
    return f"models/{model_name}/pretrained.pth"


def model_location(checkpoint_number, model_name=RUN_NAME):
    return f"models/{model_name}/epoch_{checkpoint_number}.pth"


def results_location(model_name=RUN_NAME):
    job_id = os.environ.get("JOB_ID", "local")
    return f"results/{model_name}/{job_id}"
