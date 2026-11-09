# Train-test loop for models with OOD Classes

Does giving a classifier an extra "junk" class (for out-of-distribution inputs)
make its explanations worse? This repo trains a model, fine-tunes it with a
junk class, and measures how big ReX's explanations are at every checkpoint.

## Getting started

Install the dependencies with `uv sync` (see `pyproject.toml`) or
`pip3 install -r requirements.txt`.

Then run:

```
python main.py all                      # MNIST, small CNN: runs on a laptop CPU
python -m src.analysis.visualize        # plots from the results
python -m src.analysis.sample --epoch 0 # example ReX explanations as images
```

MNIST and FashionMNIST download themselves into `./data` the first time.

On a 4-core CPU the whole MNIST run takes roughly 15-25 minutes: ~1 min per
training epoch and ~1 min of ReX per checkpoint. A GPU is used automatically
if there is one.

### Choosing the experiment

Two environment variables pick the setup (see `src/config.py`):

| Variable | Values | Meaning |
|---|---|---|
| `DATASET` | `mnist` (default), `cifar` | `cifar` is the original CIFAR-100 + ImageNet setup with a ViT. It needs the data on disk and a GPU, and exits early otherwise. |
| `OOD` | `fashion` (default), `synthetic` | MNIST only. This is the junk data. `fashion` uses FashionMNIST clothes. `synthetic` uses random strokes and blobs drawn on the fly, which look more like digits. |
| `JOB_ID` | any string (default `local`) | Puts results in `results/<run>/<JOB_ID>/`, so runs don't overwrite each other. |

For example: `OOD=synthetic JOB_ID=try2 python main.py all`.

Other useful flags: `python main.py measure --every 2` explains only every
second checkpoint (plus the first and last).

## How the pieces fit together

```
            pretrain                    train                         measure
MNIST  ──────────────▶ checkpoint 0 ──────────────▶ checkpoints 1..N ──────────▶ results/*.csv ──▶ visualize / sample
digits 0-9             (10 real        80% MNIST + 20% junk           ReX on a fixed
                        classes +       (junk all labelled 10),       subset of the
                        unused junk     one checkpoint per epoch      mixed test set
                        output)
```

1. **`pretrain`** (`src/train.py`): trains the model on the real classes only.
   The model already has a junk output (11 outputs for MNIST: digits 0-9 plus
   class 10), but no training image carries that label yet. This saves
   `models/<run>/pretrained.pth`, which `measure` treats as checkpoint 0.
2. **`train`**: loads checkpoint 0 and fine-tunes it on the *mixed* dataset
   (`src/data.py: mix_datasets`), which is 80% real images and 20% junk images,
   all labelled `cfg.ood_label`. It saves `epoch_1.pth` to `epoch_N.pth`.
3. **`measure`** (`src/analysis/evaluate.py`): for every checkpoint it takes the
   same `cfg.test_size` images from the mixed test set and runs ReX on each one.
   ReX searches for a *sufficient explanation*: the smallest set of pixels that,
   kept on their own with everything else set to 0, still gives the same
   prediction. **Explanation size** is that set's share of the image. The
   hypothesis predicts it grows once the junk class is learnt.
   - `per_class.csv`: mean/stdev size and accuracy per class, plus an `overall` row
   - `per_image.csv`: one row per image, with size, prediction, softmax entropy and P(junk)
4. **`visualize`** plots size and accuracy against checkpoint (for all images,
   and for real classes only), per-class bars, and entropy vs size.
   **`sample`** saves the input, ReX heatmap and kept pixels for a few images.

### MNIST-specific choices

- **Model** (`src/models.py`): `SmallCNN` has 3 conv layers and about 60k parameters. It reaches about 98% on MNIST after 3 epochs on a CPU.
- **No normalisation.** Pixels stay in [0, 1]. ReX hides pixels by setting them to 0, which for MNIST is the black background, so a hidden pixel just looks like empty paper. If you normalised, 0 would become mid-grey and occluded areas would look like ink.
- **ReX settings** (`cfg.rex_*`). The rex-xai defaults assume 224px images, for example a 10px minimum box. Here they are scaled down for 28x28 images.
- **FashionMNIST as junk** has the same size and greyscale format as MNIST. So unlike CIFAR (blurry 32px upscales) vs ImageNet (sharp), the model can't spot junk just from resolution.

## Original TODO

This repository was developed for CIFAR/IMAGENET and exits early if those
aren't present. The task was to refactor it to run locally with MNIST, and
perhaps to generate some OOD data for MNIST to train with. `OOD=synthetic` is
a first go at that.
