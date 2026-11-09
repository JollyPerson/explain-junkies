import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.mnist_data import MNIST_LABELS

cfg = Config()

CIFAR100_LABELS = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper",
    84: "table", 85: "tank", 86: "telephone", 87: "television", 88: "tiger",
    89: "tractor", 90: "train", 91: "trout", 92: "tulip", 93: "turtle",
    94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman",
    99: "worm",
    100: "junk",
}

LABELS = MNIST_LABELS if cfg.dataset == "mnist" else CIFAR100_LABELS


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def clean_frame(df):
    df = df.copy()
    for col in ["idx", "mean", "stdev", "accuracy", "correct", "total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_class_labels(df):
    df = df.copy()
    df["class_idx"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
    df["class_label"] = df["class_idx"].map(LABELS).fillna(df["class"].astype(str))
    return df



def collate_results(results_dir):
    frames = []

    for csv_path in Path(results_dir).rglob("per_class.csv"):
        df = pd.read_csv(csv_path)

        df["job_id"] = csv_path.parent.name
        df["model"] = csv_path.parent.parent.name

        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No per_class.csv files found in {results_dir}"
        )

    return pd.concat(frames, ignore_index=True)


def save_scatter_fit(df, x, y, out_path, title):
    plot_df = df[[x, y]].dropna().copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df["_x_num"] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])

    if plot_df.empty:
        print(f"Skipping {out_path}: no data")
        return

    if plot_df["_x_num"].isna().any():
        plot_df["_x_num"] = range(len(plot_df))

    plt.figure()
    plt.scatter(plot_df["_x_num"], plot_df[y])

    if len(plot_df) > 1:
        coeffs = np.polyfit(plot_df["_x_num"], plot_df[y], deg=1)
        fit = np.poly1d(coeffs)
        xs = plot_df["_x_num"].sort_values()
        plt.plot(xs, fit(xs))

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.xticks(plot_df["_x_num"], plot_df[x].astype(str), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def aggregate_per_class(df, idx_value=None):
    df = df[df["class"].astype(str) != "overall"].copy()
    df = add_class_labels(df)

    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    df = df.dropna(subset=["idx", "mean", "correct", "total", "class_label"])
    df = df[df["total"] > 0]

    if idx_value is not None:
        df = df[df["idx"] == float(idx_value)]

    if df.empty:
        return pd.DataFrame()

    df["weighted_mean"] = df["mean"] * df["total"]

    grouped = (
        df.groupby(["class_idx", "class_label"], as_index=False)
        .agg(
            weighted_mean=("weighted_mean", "sum"),
            correct=("correct", "sum"),
            total=("total", "sum"),
        )
    )

    grouped["exp_mean"] = grouped["weighted_mean"] / grouped["total"]
    grouped["accuracy"] = grouped["correct"] / grouped["total"]

    return grouped


def save_per_class_bar(df, out_path):
    grouped = aggregate_per_class(df)

    if grouped.empty:
        print(f"Skipping {out_path}: no per-class data")
        return

    grouped = grouped.sort_values("exp_mean", ascending=False)

    plt.figure(figsize=(max(10, len(grouped) * 0.3), 6))
    plt.bar(grouped["class_label"], grouped["exp_mean"])
    plt.xlabel("class")
    plt.ylabel("mean explanation size")
    plt.title("Explanation size per class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_top_bottom_class_lines(df, out_path):
    df = df[df["class"].astype(str) != "overall"].copy()
    df = add_class_labels(df)

    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    df = df.dropna(subset=["idx", "mean", "total", "class_label"])
    df = df[df["total"] > 0]

    if df.empty:
        print(f"Skipping {out_path}: no data")
        return

    df["weighted_mean"] = df["mean"] * df["total"]

    by_class = (
        df.groupby("class_label", as_index=False)
        .agg(
            weighted_mean=("weighted_mean", "sum"),
            total=("total", "sum"),
        )
    )
    by_class["overall_exp_mean"] = by_class["weighted_mean"] / by_class["total"]
    by_class = by_class.sort_values("overall_exp_mean", ascending=False)

    selected = pd.concat([by_class.head(2), by_class.tail(2)])["class_label"].drop_duplicates().tolist()

    plot_df = (
        df[df["class_label"].isin(selected)]
        .groupby(["idx", "class_label"], as_index=False)
        .agg(
            weighted_mean=("weighted_mean", "sum"),
            total=("total", "sum"),
        )
    )
    plot_df["exp_mean"] = plot_df["weighted_mean"] / plot_df["total"]
    plot_df = plot_df.sort_values("idx")

    plt.figure()

    for label in selected:
        rows = plot_df[plot_df["class_label"] == label]
        plt.plot(rows["idx"], rows["exp_mean"], marker="o", label=label)

    plt.xlabel("idx")
    plt.ylabel("exp_mean")
    plt.title("Top/bottom class explanation size by idx")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_per_class_bar_by_selected_idx(df, out_path, selected_idxs=None):
    df = df[df["class"].astype(str) != "overall"].copy()
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")

    available_idxs = sorted(df["idx"].dropna().unique())

    if not available_idxs:
        print(f"Skipping {out_path}: no idx values")
        return

    if selected_idxs is None:
        selected_idxs = (1, int(max(available_idxs)))

    print("Available per-class idx:", available_idxs)
    print("Plotting per-class idx:", selected_idxs)

    fig, axes = plt.subplots(
        nrows=len(selected_idxs) * 2,
        ncols=1,
        figsize=(28, 8 * len(selected_idxs)),
        sharex=False,
    )

    if len(selected_idxs) * 2 == 1:
        axes = [axes]

    for row_i, idx_value in enumerate(selected_idxs):
        exp_ax = axes[row_i * 2]
        acc_ax = axes[row_i * 2 + 1]

        grouped = aggregate_per_class(df, idx_value)

        if grouped.empty:
            exp_ax.set_title(f"idx {idx_value}: no data")
            exp_ax.axis("off")
            acc_ax.axis("off")
            continue

        grouped = grouped.sort_values("exp_mean", ascending=False)
        bar_color = "green" if idx_value == selected_idxs[-1] else None

        exp_ax.bar(grouped["class_label"], grouped["exp_mean"], color=bar_color)
        exp_ax.set_title(f"Explanation size per class — idx {idx_value}")
        exp_ax.set_xlabel("class")
        exp_ax.set_ylabel("mean explanation size")
        exp_ax.tick_params(axis="x", rotation=90)

        acc_ax.bar(grouped["class_label"], grouped["accuracy"], color=bar_color)
        acc_ax.set_title(f"Accuracy per class — idx {idx_value}")
        acc_ax.set_xlabel("class")
        acc_ax.set_ylabel("accuracy")
        acc_ax.set_ylim(0, 1)
        acc_ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_entropy_vs_size(results_dir, out_path):
    """Scatter of explanation size against softmax entropy, one dot per image.

    Uses per_image.csv. Colours separate the first and last checkpoint so you
    can see whether the junk fine-tuning moved the cloud.
    """
    frames = [pd.read_csv(p) for p in Path(results_dir).rglob("per_image.csv")]
    if not frames:
        print(f"Skipping {out_path}: no per_image.csv")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df["true_class"] != cfg.ood_label]  # real classes only
    first, last = df["idx"].min(), df["idx"].max()

    plt.figure()
    for idx_value, label in [(first, f"checkpoint {first} (pretrained)"), (last, f"checkpoint {last}")]:
        rows = df[df["idx"] == idx_value]
        plt.scatter(rows["norm_entropy"], rows["size"], s=8, alpha=0.6, label=label)
    plt.xlabel("normalised softmax entropy")
    plt.ylabel("explanation size")
    plt.title("Explanation size vs entropy (real classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    for idx_value in sorted(df["idx"].unique()):
        rows = df[df["idx"] == idx_value]
        r = rows["norm_entropy"].corr(rows["size"], method="spearman")
        print(f"checkpoint {idx_value}: spearman(entropy, size) = {r:.3f}  (n={len(rows)})")


def visualize(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = collate_results(results_dir)
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")

    is_overall = df["class"].astype(str) == "overall"
    overall = df[is_overall].sort_values(["idx", "job_id"])
    per_class = df[~is_overall]

    overall.to_csv(os.path.join(out_dir, "collated_overall.csv"), index=False)
    per_class.to_csv(os.path.join(out_dir, "collated_per_class.csv"), index=False)

    # The overall rows mix real and junk images. For the hypothesis we mostly
    # care about the real classes, so also plot their average separately.
    real = per_class[pd.to_numeric(per_class["class"], errors="coerce") != cfg.ood_label].copy()
    real["weighted"] = real["mean"] * real["total"]
    real_by_idx = real.groupby("idx", as_index=False).agg(
        weighted=("weighted", "sum"), correct=("correct", "sum"), total=("total", "sum")
    )
    real_by_idx["mean"] = real_by_idx["weighted"] / real_by_idx["total"]
    real_by_idx["accuracy"] = real_by_idx["correct"] / real_by_idx["total"]

    for frame, tag in [(overall, "all"), (real_by_idx, "real_classes")]:
        save_scatter_fit(
            frame, x="idx", y="mean",
            out_path=os.path.join(out_dir, f"mean_by_idx_{tag}.png"),
            title=f"Mean explanation size by checkpoint ({tag})",
        )
        save_scatter_fit(
            frame, x="idx", y="accuracy",
            out_path=os.path.join(out_dir, f"accuracy_by_idx_{tag}.png"),
            title=f"Accuracy by checkpoint ({tag})",
        )

    save_scatter_fit(
        overall, x="idx", y="stdev",
        out_path=os.path.join(out_dir, "stdev_by_idx.png"),
        title="Stdev of explanation size by checkpoint",
    )

    save_per_class_bar(
        per_class,
        out_path=os.path.join(out_dir, "per_class_explanation_size.png"),
    )

    save_top_bottom_class_lines(
        per_class,
        out_path=os.path.join(out_dir, "top_bottom_classes_by_idx.png"),
    )

    idxs = sorted(per_class["idx"].dropna().unique())
    save_per_class_bar_by_selected_idx(
        per_class,
        out_path=os.path.join(out_dir, "per_class_explanation_size_first_last.png"),
        selected_idxs=(idxs[0], idxs[-1]),
    )

    save_entropy_vs_size(results_dir, os.path.join(out_dir, "entropy_vs_size.png"))


if __name__ == "__main__":
    from src.utils.naming import RUN_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=f"results/{RUN_NAME}")
    parser.add_argument("--out-dir", default=f"visualizations/{RUN_NAME}")
    args = parser.parse_args()

    visualize(args.results_dir, args.out_dir)
