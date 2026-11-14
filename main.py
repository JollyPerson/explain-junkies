import argparse
import os
import time
from dataclasses import asdict

import torch
import torch.multiprocessing

from src.config import Config
from src.utils import runlog
from src.utils.naming import results_location
cfg = Config()

# avoid overloading filedescriptor count
torch.multiprocessing.set_sharing_strategy('file_system')


def run_measure(args):
    # Heavy imports (rex_xai) only when measuring.
    from src.analysis.evaluate import (
        measure, measure_parallel, stratified_subset, per_class_path, checkpoint_summary,
    )
    from src.data import get_mixed_dataset
    from src.utils.load import create_model, load_weights, checkpoint_location
    from src.utils.device import get_device

    if args.epoch:
        epochs = args.epoch
    else:
        # 0 is the pretrained model; training saves epochs 1..cfg.epochs, so
        # include the last one (the old range(cfg.epochs) stopped one short).
        # --every defaults to --save-every, so a run trained with
        # --save-every 5 is measured at 0, 5, 10, ... without a second flag.
        every = args.every or args.save_every or cfg.save_every
        epochs = list(range(0, cfg.epochs + 1, every))
        if epochs[-1] != cfg.epochs:
            epochs.append(cfg.epochs)
        never_saved = [e for e in epochs if not os.path.exists(checkpoint_location(e))]
        if never_saved:
            print(f"No checkpoint file for epochs {never_saved}, skipping them "
                  "(trained with a larger --save-every?)")
            epochs = [e for e in epochs if e not in never_saved]

    todo = [e for e in epochs if args.overwrite or not os.path.exists(per_class_path(e))]
    skipped = sorted(set(epochs) - set(todo))
    if skipped:
        print(f"Already measured, skipping: {skipped} (use --overwrite to redo)")
    if not todo:
        return

    log = runlog.get()
    # Build the dataset and the model once, not once per checkpoint.
    with log.stage("measure_load_data") as st:
        _, mix_test = get_mixed_dataset(use_parquet=args.parquet, imagenet_root=args.imagenet_root)
        data = stratified_subset(mix_test, args.per_class, seed=0)
        st.count(len(data))

    model = None
    if args.workers == 1:
        # With --workers > 1 each worker process loads its own model instead.
        model = create_model().to(get_device())

    for epoch in todo:
        # One stage per checkpoint: its time, images/s and results summary.
        with log.stage(f"measure_ep{epoch:03d}", unit="images") as st:
            if model is None:
                rows = measure_parallel(epoch, data, args.workers, precision=args.rex_precision)
            else:
                load_weights(model, epoch)
                rows = measure(model, data, str(epoch), precision=args.rex_precision)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            st.count(len(rows))
            st.set(epoch=epoch, **checkpoint_summary(rows))


def main(args):
    job_id = os.environ.get("JOB_ID", "local")
    assert job_id, "No job ID"
    train_kwargs = dict(precision=args.precision, compile_model=args.compile, resume=args.resume,
                        save_every=args.save_every or cfg.save_every)

    if args.task == "all":
        for t in ("pretrain", "train", "measure"):
            main(argparse.Namespace(**{**vars(args), "task": t}))
        return

    # Each task is one top-level stage in the run log; train.py and
    # evaluate.py add the per-epoch and per-checkpoint detail under it.
    unit = {"pretrain": "batches", "train": "batches", "measure": "images"}.get(args.task, "items")
    with runlog.get().stage(args.task, unit=unit):
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
                            help="torch.compile the model (needs Triton: Linux, or Windows with "
                                 "triton-windows installed; otherwise runs eagerly)")
    train_opts.add_argument("--resume", action="store_true",
                            help="Carry on from the last finished epoch after a crash or Ctrl+C")
    train_opts.add_argument("--save-every", type=int, default=None,
                            help="Keep weights only every N epochs, plus the last one "
                                 f"(default cfg.save_every = {cfg.save_every}). "
                                 "Applies to pretrain and train; measure follows it")

    m = parser.add_argument_group("measure")
    m.add_argument("--epoch", type=int, nargs="+",
                   help="Measure only these checkpoints, e.g. --epoch 17 for one job per checkpoint")
    m.add_argument("--every", type=int, default=None,
                   help="Measure every Nth checkpoint (the last one is always included). "
                        "Defaults to --save-every; checkpoints never saved are skipped")
    m.add_argument("--per-class", type=int, default=None,
                   help="Measure a fixed random subset of N images per class instead of all 12,500")
    m.add_argument("--rex-precision", choices=("fp32", "bf16", "fp16", "auto"), default="fp32",
                   help="Autocast dtype for ReX's forward passes. fp32 matches the original numbers")
    m.add_argument("--workers", type=int, default=1,
                   help="Explain images in N processes sharing the GPU. ReX is mostly "
                        "single-threaded Python, so 4-6 workers run several times faster")
    m.add_argument("--overwrite", action="store_true",
                   help="Re-measure checkpoints that already have results")

    lg = parser.add_argument_group("logging")
    lg.add_argument("--log-interval", type=float, default=10.0,
                    help="Seconds between GPU/CPU/RAM samples in resources.csv (default 10)")
    lg.add_argument("--no-log", action="store_true",
                    help="Don't write the run log (results and checkpoints are still saved)")
    args = parser.parse_args(argv)
    for name in ("every", "save_every", "workers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be 1 or more")
    return args


def log_dir(args):
    """results/<model>/<job_id>/logs/<UTC start>_<task>[_epN], one folder per
    invocation so parallel measure containers never share a file."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{stamp}_{args.task}"
    if args.task == "measure" and args.epoch:
        name += "_ep" + "-".join(str(e) for e in args.epoch)
    return os.path.join(results_location(), "logs", name)


if __name__ == "__main__":
    args = parse_args()
    if args.no_log:
        main(args)
    else:
        log = runlog.start(args.task, log_dir(args), interval=args.log_interval,
                           config=asdict(cfg), args=vars(args))
        try:
            main(args)
        except BaseException as e:
            # KeyboardInterrupt and Modal's timeout land here too, so the
            # log still says how far the run got and why it stopped.
            log.close(f"failed: {type(e).__name__}: {e}"[:300])
            raise
        log.close("ok")
