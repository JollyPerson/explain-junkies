"""Run the CIFAR-100 + ImageNet-junk pipeline (ViT) on a Modal GPU.

One-time setup on your own machine:
    pip install modal
    modal setup                                   # logs in through the browser

    # Put the junk cache on the Modal volume (about 2.3 GB, once):
    rclone copy r2:ooddata/ooddata/junk_cache ./junk_cache -P
    modal volume put ood-data ./junk_cache data/junk_cache

Run (from the repo root, on this branch):
    # 2-minute smoke test first
    modal run modal_app.py --job-id smoke --smoke --args "--save-every 2"

    # the real run; --detach keeps it going after you close the laptop
    modal run --detach modal_app.py --job-id vit1 --args "--save-every 5 --per-class 50"

    # if it hits the timeout or crashes, carry on from the last finished epoch
    modal run --detach modal_app.py --job-id vit1 --task train --args "--save-every 5 --resume"

Measure (ReX) on cheap L4s, one container per checkpoint, all at once.
Training must have finished; checkpoints are read from the volume, and any
checkpoint that already has results is skipped:
    modal run --detach modal_app.py --job-id vit1 --task measure-all --args "--per-class 50 --workers 6"
    # only some checkpoints:  add --epochs 0,30,60

Get the results back:
    modal volume ls ood-data results
    modal volume get ood-data results ./results

CIFAR-100 downloads itself onto the volume the first time. With the junk
cache in data/junk_cache, ImageNet itself is never needed.
"""

import os
import shlex
import subprocess

import modal

# Change this to pick the GPU: "L4" (cheapest), "L40S", "A100-80GB" or "H100".
GPU = "L40S"
# ReX is mostly single-threaded Python, so a big GPU sits idle while it runs.
# measure_all runs it on L4s instead, with several worker processes each.
MEASURE_GPU = "L4"

APP_DIR = "/root/ood"
VOL_DIR = "/vol"

app = modal.App("ood-cifar")

# Python 3.12 because rex-xai pins numpy 1.26.4, which has no 3.13 wheels.
# PyPI's Linux torch wheels include CUDA, so the cu126 index isn't needed.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "timm",
        "rex-xai==0.4.1",
        "numpy==1.26.4",
        "pandas",
        "pyarrow>=16,<20",
        "matplotlib",
        "pillow",
        "tqdm",
    )
    # Copy the code in; data, checkpoints and results live on the volume.
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=[".venv", ".git", "**/__pycache__", "data", "models", "results",
                "logs", "imagenet", "junk_cache"],
    )
)

vol = modal.Volume.from_name("ood-data", create_if_missing=True)


def _link_volume():
    os.chdir(APP_DIR)
    # main.py reads and writes ./data, ./models and ./results. Point them at
    # the volume so the files outlive the container.
    for name in ("data", "models", "results"):
        os.makedirs(f"{VOL_DIR}/{name}", exist_ok=True)
        if not os.path.exists(name):
            os.symlink(f"{VOL_DIR}/{name}", name)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,                  # matches cfg.num_workers = 8 data-loading processes
    memory=32768,           # MB
    volumes={VOL_DIR: vol},
    timeout=24 * 60 * 60,   # seconds; Modal's maximum
)
def run(job_id: str, task: str, args: str, smoke: bool):
    _link_volume()

    cache = "data/junk_cache/train_images.npy"
    print("junk cache found" if os.path.exists(cache) else
          f"WARNING: no {cache} on the volume; the run will look for ImageNet instead")
    subprocess.run(["nvidia-smi"], check=False)

    env = {**os.environ, "JOB_ID": job_id}
    if smoke:
        env["SMOKE"] = "1"
    cmd = ["python", "main.py", task, *shlex.split(args)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        # Save whatever was written, even if the run crashed or timed out.
        vol.commit()


@app.function(
    image=image,
    gpu=MEASURE_GPU,
    cpu=8,                  # one core per ReX worker process, plus spare
    memory=32768,           # MB; each worker holds its own model and data copy
    volumes={VOL_DIR: vol},
    timeout=24 * 60 * 60,
)
def measure_one(job_id: str, epoch: int, args: str, smoke: bool):
    """ReX for a single checkpoint. measure_all starts one of these per checkpoint."""
    _link_volume()
    env = {**os.environ, "JOB_ID": job_id}
    if smoke:
        env["SMOKE"] = "1"
    cmd = ["python", "main.py", "measure", "--epoch", str(epoch), *shlex.split(args)]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        vol.commit()


def _measure_all(job_id, epochs, args, smoke):
    """Start one L4 container per checkpoint, all running at the same time."""
    todo = [int(e) for e in epochs.split(",") if e.strip()]
    print(f"Measuring checkpoints {todo} on {MEASURE_GPU}, one container each, args={args!r}")
    calls = {e: measure_one.spawn(job_id, e, args, smoke) for e in todo}
    failed = []
    for e, call in calls.items():
        try:
            call.get()
            print(f"checkpoint {e}: done")
        except Exception as err:  # keep waiting for the others
            print(f"checkpoint {e}: FAILED ({err})")
            failed.append(e)
    if failed:
        print(f"Rerun just these with --epochs {','.join(map(str, failed))}")


@app.local_entrypoint()
def main(job_id: str = "modal", task: str = "all", args: str = "", smoke: bool = False,
         epochs: str = "0,5,10,15,20,25,30,35,40,45,50,55,60"):
    if task == "measure-all":
        _measure_all(job_id, epochs, args, smoke)
    else:
        print(f"Starting task={task} job_id={job_id} gpu={GPU} smoke={smoke} args={args!r}")
        run.remote(job_id, task, args, smoke)
    print("Done. Fetch results with: modal volume get ood-data results ./results")
