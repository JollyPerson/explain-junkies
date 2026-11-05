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

def main(task):
    job_id = os.environ.get("JOB_ID", "local")
    assert job_id, "No job ID"

    os.makedirs(f"models/{cfg.model}", exist_ok=True)

    match task:
        case "pretrain": pretrain()
        case "train": train()
        case "measure":
            for i in range(cfg.epochs):
                _, mix_test = get_mixed_dataset()
                
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
                main(t)


if __name__ == "__main__":
    main(sys.argv[1])
