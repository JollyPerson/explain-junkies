"""Analyse one ReX measure run: does explanation size grow with junk training?

Usage (from anywhere, needs pandas + matplotlib):
    python analyse_run.py path/to/results/vit_base_patch16_224/vit1

The folder must contain per_image_epoch_NNN.csv files written by
`main.py measure`. Each row is one test image at one checkpoint:
    idx         checkpoint (0 = pretrained, before any junk training)
    image       position in the fixed image subset (same image at every checkpoint)
    true_class  0-99 = CIFAR-100 class, 100 = junk (ImageNet)
    pred_class  what the model predicted
    correct     1 if pred_class == true_class
    confidence  model's probability for pred_class
    size        fraction of pixels in ReX's sufficient explanation (0-1);
                bigger = the model needs more of the image = less explainable.
                NaN = ReX found no explanation for this image.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JUNK = 100  # cfg.ood_label for CIFAR; use 10 for the MNIST runs


# 1. Load every checkpoint into one table ------------------------------------
folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = sorted(folder.glob("per_image_epoch_*.csv"))
if not files:
    sys.exit(f"No per_image_epoch_*.csv files in {folder}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["group"] = np.where(df["true_class"] == JUNK, "junk", "real")
df["pred_junk"] = (df["pred_class"] == JUNK).astype(int)
print(f"{len(files)} checkpoints, {df['image'].nunique()} images each, "
      f"{(df['group'] == 'junk').sum() // len(files)} of them junk\n")


# 2. Headline table: one row per checkpoint ----------------------------------
# Real and junk images are reported separately. The overall average can move
# just because the junk images change (that is what happened on MNIST), so
# the hypothesis is about the REAL images.
def summarise(g):
    s = g["size"].dropna()
    return pd.Series({
        "mean_size": s.mean(),
        "sem": s.std(ddof=1) / np.sqrt(len(s)),   # standard error of the mean
        "accuracy": g["correct"].mean(),
        "pred_as_junk": g["pred_junk"].mean(),
        "no_explanation": g["size"].isna().sum(),
        "n": len(g),
    })

table = df.groupby(["idx", "group"]).apply(summarise).unstack("group")
pd.set_option("display.width", 160)
print("Per checkpoint (real = CIFAR images, junk = ImageNet images):")
print(table.round(4).to_string(), "\n")


# 3. Paired test: same image, first vs last checkpoint ------------------------
# Every checkpoint explains the same images, so compare each image with
# itself. This removes image-to-image variation and is much more sensitive
# than comparing two averages.
first, last = df["idx"].min(), df["idx"].max()
wide = df.pivot_table(index=["image", "group"], columns="idx", values="size").dropna()
for grp in ("real", "junk"):
    d = (wide[last] - wide[first]).xs(grp, level="group")
    lo, hi = d.mean() - 1.96 * d.sem(), d.mean() + 1.96 * d.sem()
    grew = (d > 0).mean()
    print(f"{grp:4s}: size change {first}->{last} = {d.mean():+.4f} "
          f"(95% CI {lo:+.4f} to {hi:+.4f}), n={len(d)}, "
          f"{grew:.0%} of images got bigger explanations")
print("If the CI excludes 0 for 'real', the change is unlikely to be noise.\n")


# 4. Plots --------------------------------------------------------------------
out = folder / "plots"
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for grp, colour in (("real", "tab:blue"), ("junk", "tab:orange")):
    m = table[("mean_size", grp)]
    e = table[("sem", grp)]
    axes[0].errorbar(m.index, m, yerr=1.96 * e, marker="o", capsize=3,
                     color=colour, label=grp)
    axes[1].plot(table.index, table[("accuracy", grp)], marker="o",
                 color=colour, label=grp)
axes[0].set(title="Explanation size (95% CI)", xlabel="checkpoint (epochs of junk training)",
            ylabel="fraction of image")
axes[1].set(title="Accuracy", xlabel="checkpoint", ylabel="accuracy")

# Paired view: every real image's size at the first vs the last checkpoint.
real = wide.xs("real", level="group")
axes[2].scatter(real[first], real[last], s=6, alpha=0.4)
lim = [0, max(real[first].max(), real[last].max())]
axes[2].plot(lim, lim, "k--", lw=1, label="no change")
axes[2].set(title=f"Real images: checkpoint {first} vs {last}",
            xlabel=f"size at {first}", ylabel=f"size at {last}")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(out / "summary.png", dpi=150)

# Does confidence explain size? (the entropy workstream idea)
fig, ax = plt.subplots(figsize=(5, 4))
sub = df[(df["group"] == "real") & df["size"].notna()]
ax.scatter(sub["confidence"], sub["size"], s=4, alpha=0.2)
ax.set(title="Real images, all checkpoints", xlabel="model confidence", ylabel="explanation size")
fig.tight_layout()
fig.savefig(out / "confidence_vs_size.png", dpi=150)
print(f"Correlation confidence vs size (real): {sub['confidence'].corr(sub['size']):+.3f}")

table.to_csv(out / "summary_table.csv")
print(f"\nPlots and table saved in {out}")
