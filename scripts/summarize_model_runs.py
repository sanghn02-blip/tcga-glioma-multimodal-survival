#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd


MODEL_META = {
    "clinical": ("Clinical", "Clinical/covariates"),
    "rna": ("RNA", "selected RNA genes"),
    "rna_clinical": ("RNA + Clinical", "selected RNA genes + clinical/covariates"),
    "wsi": ("WSI", "WSI handcrafted features"),
    "wsi_clinical": ("WSI + Clinical", "WSI handcrafted features + clinical/covariates"),
    "wsi_rna": ("WSI + RNA", "WSI handcrafted features + selected RNA genes"),
    "wsi_rna_clinical": (
        "WSI + RNA + Clinical",
        "WSI handcrafted features + selected RNA genes + clinical/covariates",
    ),
}


def model_meta_for_run(feature_set, metrics_path):
    model_name, input_label = MODEL_META.get(feature_set, (feature_set, feature_set))
    run_dir_name = metrics_path.parent.name.lower()
    if "wsi_embedding" not in run_dir_name:
        return model_name, input_label
    encoder_label = "ResNet18"
    if "uni" in run_dir_name:
        encoder_label = "UNI"
    elif "conch" in run_dir_name:
        encoder_label = "CONCH"
    if feature_set == "wsi":
        return f"WSI {encoder_label} Embedding", f"{encoder_label} patch embeddings"
    if feature_set == "wsi_clinical":
        return f"WSI {encoder_label} Embedding + Clinical", f"{encoder_label} patch embeddings + clinical/covariates"
    if feature_set == "wsi_rna":
        return f"WSI {encoder_label} Embedding + RNA", f"{encoder_label} patch embeddings + selected RNA genes"
    if feature_set == "wsi_rna_clinical":
        return (
            f"WSI {encoder_label} Embedding + RNA + Clinical",
            f"{encoder_label} patch embeddings + selected RNA genes + clinical/covariates",
        )
    return model_name, input_label


def read_metrics(path):
    metrics = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2:
                metrics[row[0]] = row[1]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/model_comparison.tsv"))
    parser.add_argument("--run-dirs", nargs="*", type=Path)
    parser.add_argument("--cohort-label")
    args = parser.parse_args()

    rows = []
    if args.run_dirs:
        metric_paths = [run_dir / "cox_metrics.tsv" for run_dir in args.run_dirs]
    else:
        metric_paths = sorted(args.results_root.glob("*/cox_metrics.tsv"))
    for metrics_path in metric_paths:
        if not metrics_path.exists():
            continue
        metrics = read_metrics(metrics_path)
        feature_set = metrics.get("feature_set", metrics_path.parent.name)
        model_name, input_label = model_meta_for_run(feature_set, metrics_path)
        n_rna = metrics.get("n_rna_features_requested", "0")
        if "selected RNA genes" in input_label:
            input_label = input_label.replace("selected RNA genes", f"{n_rna} selected RNA genes")
        n_wsi_requested = metrics.get("n_wsi_features_requested", "0")
        rows.append(
            {
                "cohort": args.cohort_label or "",
                "model": model_name,
                "input": input_label,
                "run_dir": str(metrics_path.parent),
                "feature_set": feature_set,
                "n_cases": metrics.get("n_cases"),
                "folds": metrics.get("folds"),
                "c_index_cv": metrics.get("c_index_cv"),
                "logrank_p": metrics.get("logrank_p"),
                "n_clinical_features": metrics.get("n_clinical_features"),
                "n_rna_features_requested": n_rna,
                "n_wsi_features": metrics.get("n_wsi_features", "0"),
                "n_wsi_features_requested": n_wsi_requested,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["c_index_cv"] = pd.to_numeric(frame["c_index_cv"], errors="coerce")
        frame = frame.sort_values("c_index_cv", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {len(frame)} model rows to {args.out}")


if __name__ == "__main__":
    main()
