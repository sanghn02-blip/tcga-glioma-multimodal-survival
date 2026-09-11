#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


def read_metric_table(path):
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t", header=None, names=["key", "value"])
    return dict(zip(frame["key"].astype(str), frame["value"]))


def maybe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value, default=0):
    number = maybe_float(value)
    if number is None:
        return default
    return int(number)


def log_existing_artifacts(mlflow, paths, artifact_path):
    for path in paths:
        path = ROOT / path
        if path.exists() and path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def should_skip(run_name, existing_run_names, allow_duplicates):
    return not allow_duplicates and run_name in existing_run_names


def log_baseline_runs(mlflow, comparison_path, existing_run_names, allow_duplicates):
    comparison = pd.read_csv(comparison_path, sep="\t")
    runs_logged = 0
    runs_skipped = 0
    for _, row in comparison.iterrows():
        run_dir = Path(row["run_dir"])
        run_name = f"{row['cohort']} | {row['model']}"
        if should_skip(run_name, existing_run_names, allow_duplicates):
            runs_skipped += 1
            continue
        metrics = read_metric_table(ROOT / run_dir / "cox_metrics.tsv")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "project": "TCGA-LGG/GBM",
                    "stage": "baseline_survival",
                    "cohort": row["cohort"],
                    "model": row["model"],
                    "input": row["input"],
                    "run_dir": str(run_dir),
                }
            )
            mlflow.log_params(
                {
                    "n_cases": safe_int(row["n_cases"]),
                    "folds": safe_int(row["folds"]),
                    "n_rna_features_requested": safe_int(row.get("n_rna_features_requested", 0)),
                    "n_clinical_features": safe_int(row.get("n_clinical_features", 0)),
                    "n_wsi_features": safe_int(row.get("n_wsi_features", 0)),
                    "n_wsi_features_requested": safe_int(row.get("n_wsi_features_requested", 0)),
                    "feature_set": metrics.get("feature_set", safe_name(row["model"])),
                    "penalizer": metrics.get("penalizer", ""),
                    "l1_ratio": metrics.get("l1_ratio", ""),
                }
            )
            for key in ["c_index_cv", "logrank_p", "median_risk"]:
                value = maybe_float(metrics.get(key, row.get(key)))
                if value is not None:
                    mlflow.log_metric(key, value)
            log_existing_artifacts(
                mlflow,
                [
                    run_dir / "cox_metrics.tsv",
                    run_dir / "cox_predictions.tsv",
                    run_dir / "cox_km_curve.tsv",
                    run_dir / "selected_features.tsv",
                ],
                artifact_path="survival_outputs",
            )
            runs_logged += 1
            existing_run_names.add(run_name)
    return runs_logged, runs_skipped


def log_interpretability_run(mlflow, existing_run_names, allow_duplicates):
    run_name = "RNA interpretability summary"
    if should_skip(run_name, existing_run_names, allow_duplicates):
        return 0, 1
    top_genes = ROOT / "results/rna_interpretability_top_genes.tsv"
    if not top_genes.exists():
        return 0, 0
    frame = pd.read_csv(top_genes, sep="\t")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "TCGA-LGG/GBM",
                "stage": "rna_interpretability",
                "method": "cox_coefficients_and_patient_contributions",
            }
        )
        mlflow.log_param("top_gene_rows", len(frame))
        mlflow.log_param("cohorts", ",".join(sorted(frame["cohort"].dropna().unique())))
        mlflow.log_artifact(str(top_genes), artifact_path="interpretability")
        figures_dir = ROOT / "results/figures"
        if figures_dir.exists():
            for figure in sorted(figures_dir.glob("rna_top_genes_*.png")):
                mlflow.log_artifact(str(figure), artifact_path="interpretability/figures")
    existing_run_names.add(run_name)
    return 1, 0


def log_wsi_preprocessing_run(mlflow, existing_run_names, allow_duplicates, dataset_key, dataset_dir, max_preview_files):
    run_name = f"WSI {dataset_key} preprocessing"
    if should_skip(run_name, existing_run_names, allow_duplicates):
        return 0, 1
    dataset_root = ROOT / dataset_dir
    summary = dataset_root / "slide_summary.tsv"
    patch_index = dataset_root / "patch_index.tsv"
    patch_images = dataset_root / "patch_images/patch_image_index.tsv"
    handcrafted_features = dataset_root / "wsi_handcrafted_features.tsv"
    if not summary.exists():
        return 0, 0
    slides = pd.read_csv(summary, sep="\t")
    patches = pd.read_csv(patch_index, sep="\t") if patch_index.exists() else pd.DataFrame()
    images = pd.read_csv(patch_images, sep="\t") if patch_images.exists() else pd.DataFrame()
    features = pd.read_csv(handcrafted_features, sep="\t") if handcrafted_features.exists() else pd.DataFrame()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "TCGA-LGG/GBM",
                "stage": "wsi_preprocessing",
                "dataset": dataset_key,
                "method": "tissue_detection_patch_index",
            }
        )
        mlflow.log_params(
            {
                "slides": len(slides),
                "slides_with_patches": int((slides["patches"] > 0).sum()) if "patches" in slides else 0,
                "candidate_patches": len(patches),
                "patch_images": len(images),
                "handcrafted_feature_rows": len(features),
                "handcrafted_feature_columns": features.shape[1] if not features.empty else 0,
            }
        )
        for path in [summary, patch_index, patch_images, handcrafted_features]:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=f"wsi_{dataset_key}")
        for overlay in sorted(dataset_root.glob("*/tissue_overlay.jpg"))[:max_preview_files]:
            mlflow.log_artifact(str(overlay), artifact_path=f"wsi_{dataset_key}/overlays")
        for patch in sorted((dataset_root / "patch_images").glob("*/*.jpg"))[: max_preview_files * 2]:
            mlflow.log_artifact(str(patch), artifact_path=f"wsi_{dataset_key}/patch_examples")
    existing_run_names.add(run_name)
    return 1, 0


def log_wsi_feature_run(mlflow, existing_run_names, allow_duplicates, dataset_key, dataset_dir):
    run_name = f"WSI {dataset_key} handcrafted feature extraction"
    if should_skip(run_name, existing_run_names, allow_duplicates):
        return 0, 1
    features_path = ROOT / dataset_dir / "wsi_handcrafted_features.tsv"
    if not features_path.exists():
        return 0, 0
    features = pd.read_csv(features_path, sep="\t")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "TCGA-LGG/GBM",
                "stage": "wsi_feature_extraction",
                "dataset": dataset_key,
                "method": "handcrafted_color_tissue_summary",
            }
        )
        mlflow.log_params(
            {
                "slides": len(features),
                "feature_columns": features.shape[1],
                "mean_patches_used": float(features["wsi_patches_used"].mean()) if "wsi_patches_used" in features else 0,
            }
        )
        mlflow.log_artifact(str(features_path), artifact_path="wsi_features")
    existing_run_names.add(run_name)
    return 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=Path("results/stratified_model_comparison.tsv"))
    parser.add_argument("--experiment", default="TCGA_Glioma_Survival")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("MLflow is not installed. Run: .venv/bin/python -m pip install -r requirements-mlflow.txt") from exc

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    existing_run_names = {
        run.info.run_name
        for run in client.search_runs([experiment.experiment_id], max_results=5000)
    }

    runs = 0
    skipped = 0
    comparison_path = ROOT / args.comparison
    if comparison_path.exists():
        logged_now, skipped_now = log_baseline_runs(
            mlflow,
            comparison_path,
            existing_run_names,
            args.allow_duplicates,
        )
        runs += logged_now
        skipped += skipped_now
        run_name = "Model comparison table"
        if should_skip(run_name, existing_run_names, args.allow_duplicates):
            skipped += 1
        else:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"project": "TCGA-LGG/GBM", "stage": "model_comparison_summary"})
                mlflow.log_artifact(str(comparison_path), artifact_path="model_comparison")
            existing_run_names.add(run_name)
            runs += 1
    logged_now, skipped_now = log_interpretability_run(mlflow, existing_run_names, args.allow_duplicates)
    runs += logged_now
    skipped += skipped_now
    for dataset_key, dataset_dir, max_preview_files in [
        ("smoke", Path("data/processed/wsi_smoke"), 6),
        ("dev50", Path("data/processed/wsi_dev_50"), 8),
    ]:
        logged_now, skipped_now = log_wsi_preprocessing_run(
            mlflow,
            existing_run_names,
            args.allow_duplicates,
            dataset_key,
            dataset_dir,
            max_preview_files,
        )
        runs += logged_now
        skipped += skipped_now
        logged_now, skipped_now = log_wsi_feature_run(
            mlflow,
            existing_run_names,
            args.allow_duplicates,
            dataset_key,
            dataset_dir,
        )
        runs += logged_now
        skipped += skipped_now
    print(f"logged {runs} experiment runs to {args.tracking_uri}; skipped {skipped} existing runs")


if __name__ == "__main__":
    main()
