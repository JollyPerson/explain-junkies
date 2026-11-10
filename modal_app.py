"""Run the MNIST junk-class pipeline on a Modal GPU.

One-time setup on your own machine:
    pip install modal
    modal setup                     # logs in through the browser

Run (from the repo root):
    modal run modal_app.py --job-id test1                 # watch it live
    modal run --detach modal_app.py --job-id gpu1         # keeps going after you close the laptop
    modal run modal_app.py --job-id synth1 --ood synthetic

Get the results back:
    modal volume ls ood-data results
    modal volume get ood-data results ./results
    modal volume get ood-data models ./models

Everything the code writes (MNIST downloads, checkpoints, CSVs) goes to the
"ood-data" Modal volume, so it survives after the container stops and later
runs reuse the downloaded datasets.
"""

import os
import subprocess

import modal

APP_DIR = "/root/ood"
VOL_DIR = "/vol"

app = modal.App("ood")

# The container: Python 3.12 (rex-xai pins numpy 1.26.4, which has no 3.13
# wheels) plus the same packages as pyproject.toml. PyPI's Linux torch wheels
# already include CUDA, so the cu126 index isn't needed here. Built once and cached.
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
    # Copy the code in, leaving out local data, checkpoints and results:
    # those live on the volume instead.
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=[".venv", ".git", "**/__pycache__", "data", "models", "results", "logs"],
    )
)

vol = modal.Volume.from_name("ood-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",              # cheap and plenty for MNIST; "A100" or "H100" for the ViT
    cpu=4,                 # Modal's default is a fraction of a core, and ReX on MNIST is CPU-bound
    memory=8192,           # MB
    volumes={VOL_DIR: vol},
    timeout=2 * 60 * 60,   # seconds; the MNIST run should take well under an hour
)
def run(job_id: str, ood: str, task: str):
    os.chdir(APP_DIR)

    # main.py writes to ./data, ./models and ./results. Point those at the
    # volume so the files outlive the container.
    for name in ("data", "models", "results"):
        os.makedirs(f"{VOL_DIR}/{name}", exist_ok=True)
        if not os.path.exists(name):
            os.symlink(f"{VOL_DIR}/{name}", name)

    subprocess.run(["nvidia-smi"], check=False)

    env = {**os.environ, "DATASET": "mnist", "OOD": ood, "JOB_ID": job_id}
    try:
        subprocess.run(["python", "main.py", task], env=env, check=True)
    finally:
        # Save whatever was written, even if the run crashed halfway.
        vol.commit()


@app.local_entrypoint()
def main(job_id: str = "modal", ood: str = "fashion", task: str = "all"):
    print(f"Starting task={task} ood={ood} job_id={job_id} on Modal")
    run.remote(job_id, ood, task)
    print(f"Done. Fetch results with: modal volume get ood-data results ./results")
