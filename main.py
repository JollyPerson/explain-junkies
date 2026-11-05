print("test")
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

def main(task, use_parquet=False, imagenet_root=None):
    print("test")
    job_id = os.environ.get("JOB_ID", "local")
    print(job_id)
    assert job_id, "No job ID"

    match task:
        case "pretrain": pretrain()
        case "train":
            train(use_parquet=use_parquet, imagenet_root=imagenet_root)
        case "measure":
            for i in range(cfg.epochs):
                _, mix_test = get_mixed_dataset(
                    use_parquet=use_parquet,
                    imagenet_root=imagenet_root,
                )
                
                model = load_model(epoch=i)
                measure(model, mix_test, str(i))

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        case "all":
            for t in [
                "pretrain",
                "train",
                "measure"
            ]:
                main(t, use_parquet=use_parquet, imagenet_root=imagenet_root)


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
    args = parser.parse_args()
    main(args.task, use_parquet=args.parquet, imagenet_root=args.imagenet_root)
