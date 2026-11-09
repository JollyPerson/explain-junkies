import argparse
import sys
import os

import torch
import torch.multiprocessing

from src.utils.load import load_model
from src.train import pretrain, train
from src.analysis.evaluate import measure
from src.config import Config
from src.data import get_mixed_dataset


# avoid overloading filedescriptor count
torch.multiprocessing.set_sharing_strategy('file_system')

cfg = Config()

def checkpoints(every=1):
    """0, every, 2*every, ..., always ending with the last epoch."""
    eps = list(range(0, cfg.epochs + 1, every))
    if eps[-1] != cfg.epochs:
        eps.append(cfg.epochs)
    return eps


def main(task, use_parquet=False, imagenet_root=None, every=1):
    job_id = os.environ.get("JOB_ID", "local")
    assert job_id, "No job ID"
    print(f"task={task}  dataset={cfg.dataset}  junk={cfg.ood_source}  model={cfg.model}  job={job_id}")


    match task:
        case "pretrain": pretrain()
        case "train":
            train(use_parquet=use_parquet, imagenet_root=imagenet_root)
        case "measure":
            # Load the test data once; it is the same for every checkpoint.
            _, mix_test = get_mixed_dataset(
                use_parquet=use_parquet,
                imagenet_root=imagenet_root,
            )
            # Checkpoint 0 is the pretrained model (no junk class yet);
            # 1..epochs are the junk fine-tuning epochs.
            for i in checkpoints(every):
                model = load_model(epoch=i)
                measure(model, mix_test, idx=str(i))

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        case "all":
            for t in [
                "pretrain",
                "train",
                "measure"
            ]:
                main(t, use_parquet=use_parquet, imagenet_root=imagenet_root, every=every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("pretrain", "train", "measure", "all"))
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Use ImageNet parquet shards instead of the torchvision ImageNet files",
    )
    parser.add_argument(
        "--imagenet-root",
        default=None,
        help="Directory containing ImageNet files or parquet shards",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="measure: only explain every Nth checkpoint (plus the first and last)",
    )
    args = parser.parse_args()
    main(args.task, use_parquet=args.parquet, imagenet_root=args.imagenet_root, every=args.every)
