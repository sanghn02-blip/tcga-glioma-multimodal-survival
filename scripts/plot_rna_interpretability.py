#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_top_genes(frame, title, out_path, top_n):
    data = frame.sort_values("coef", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    data = data.sort_values("coef")
    colors = ["#276fbf" if value < 0 else "#c44536" for value in data["coef"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(data["gene"], data["coef"], color=colors)
    ax.axvline(0, color="#303030", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Cox coefficient; positive means higher predicted risk")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-genes", type=Path, default=Path("results/rna_interpretability_top_genes.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    top = pd.read_csv(args.top_genes, sep="\t")
    adjusted = top[top["analysis"].eq("RNA adjusted for clinical")].copy()
    if adjusted.empty:
        raise ValueError("No 'RNA adjusted for clinical' rows found.")

    for cohort, cohort_frame in adjusted.groupby("cohort", sort=False):
        slug = cohort.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        out_path = args.out_dir / f"rna_top_genes_{slug}_clinical_adjusted.png"
        plot_top_genes(cohort_frame, f"{cohort}: clinical-adjusted RNA risk genes", out_path, args.top_n)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
