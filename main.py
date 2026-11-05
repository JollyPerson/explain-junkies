import argparse
import os

import torch
import torch.multiprocessing

from src.config import Config
cfg = Config()

# avoid overloading filedescriptor count
torch.multiprocessing.set_sharing_strategy('file_system')


def run_measure(args):
    # Heavy imports (rex_xai) only when measuring.
    from src.analysis.evaluate import measure, stratified_subset, per_class_path
    from src.data import get_mixed_dataset
    from src.utils.load import create_model, load_weights
    from src.utils.device import get_device

    if args.epoch:
        epochs = args.epoch
    else:
        # 0 is the pretrained model; training saves epochs 1..cfg.epochs, so
        # include the last one (the old range(cfg.epochs) stopped one short).
        epochs = list(range(0, cfg.epochs + 1, args.every))
        if epochs[-1] != cfg.epochs:
            epochs.append(cfg.epochs)

    todo = [e for e in epochs if args.overwrite or not os.path.exists(per_class_path(e))]
    skipped = sorted(set(epochs) - set(todo))
    if skipped:
        print(f"Already measured, skipping: {skipped} (use --overwrite to redo)")
    if not todo:
        return

    # Build the dataset and the model once, not once per checkpoint.
    _, mix_test = get_mixed_dataset(use_parquet=args.parquet, imagenet_root=args.imagenet_root)
    data = stratified_subset(mix_test, args.per_class, seed=0)

    model = create_model().to(get_device())
    for epoch in todo:
        load_weights(model, epoch)
        measure(model, data, str(epoch), precision=args.rex_precision)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(args):
    job_id = os.environ.get("JOB_ID", "local")
    assert job_id, "No job ID"
    train_kwargs = dict(precision=args.precision, compile_model=args.compile)

    match args.task:
        case "prepare":
            from src.data import build_junk_cache
            build_junk_cache(imagenet_root=args.imagenet_root, use_parquet=args.parquet)
        case "pretrain":
            from src.train import pretrain
            pretrain(**train_kwargs)
        case "train":
            from src.train import train
            train(use_parquet=args.parquet, imagenet_root=args.imagenet_root, **train_kwargs)
        case "measure":
            run_measure(args)
        case "all":
            for t in ("pretrain", "train", "measure"):
                main(argparse.Namespace(**{**vars(args), "task": t}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("prepare", "pretrain", "train", "measure", "all"))
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

    train_opts = parser.add_argument_group("training")
    train_opts.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
                            help="Autocast dtype for training. auto = bf16 if the GPU supports it, else fp16")
    train_opts.add_argument("--compile", action="store_true",
                            help="torch.compile the model (needs a Volta or newer GPU)")

    m = parser.add_argument_group("measure")
    m.add_argument("--epoch", type=int, nargs="+",
                   help="Measure only these checkpoints, e.g. --epoch 17 for one job per checkpoint")
    m.add_argument("--every", type=int, default=1,
                   help="Measure every Nth checkpoint (the last one is always included)")
    m.add_argument("--per-class", type=int, default=None,
                   help="Measure a fixed random subset of N images per class instead of all 12,500")
    m.add_argument("--rex-precision", choices=("fp32", "bf16", "fp16", "auto"), default="fp32",
                   help="Autocast dtype for ReX's forward passes. fp32 matches the original numbers")
    m.add_argument("--overwrite", action="store_true",
                   help="Re-measure checkpoints that already have results")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
