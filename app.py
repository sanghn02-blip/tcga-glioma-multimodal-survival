#!/usr/bin/env python3
import argparse
import io
import json
import mimetypes
import subprocess
import sys
import tempfile
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = ROOT / "models/rna_clinical_final"


RUNS = {
    "all_clinical": "results/full_clinical_baseline",
    "all_rna": "results/full_rna_baseline_fast",
    "all_rna_clinical": "results/full_rna_clinical_baseline_fast",
    "lgg_clinical": "results/stratified_tcga_lgg_clinical",
    "lgg_rna": "results/stratified_tcga_lgg_rna_fast",
    "lgg_rna_clinical": "results/stratified_tcga_lgg_rna_clinical_fast",
    "gbm_clinical": "results/stratified_tcga_gbm_clinical",
    "gbm_rna": "results/stratified_tcga_gbm_rna_fast",
    "gbm_rna_clinical": "results/stratified_tcga_gbm_rna_clinical_fast",
}

WSI_DATASETS = {
    "dev50": {
        "label": "WSI dev 50",
        "processed_dir": "data/processed/wsi_dev_50",
    },
    "smoke": {
        "label": "WSI smoke",
        "processed_dir": "data/processed/wsi_smoke",
    },
}


def read_tsv(path):
    path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def read_key_value(path):
    path = ROOT / path
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t", header=None, names=["key", "value"])
    return dict(zip(frame["key"], frame["value"]))


def clean_records(frame):
    if frame.empty:
        return []
    return json.loads(frame.where(pd.notnull(frame), None).to_json(orient="records"))


def read_uploaded_table(data, filename):
    suffix = Path(filename or "").suffix.lower()
    buffer = io.BytesIO(data)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(buffer)
        except ImportError as exc:
            raise ValueError("Excel .xlsx 입력은 openpyxl 설치 후 사용할 수 있습니다. 지금은 CSV/TSV 파일을 사용하세요.") from exc
    return pd.read_csv(buffer, sep="\t")


def preview_rna_upload(data, filename, limit=12):
    frame = read_uploaded_table(data, filename)
    if frame.empty:
        raise ValueError("RNA 파일에 읽을 수 있는 행이 없습니다.")
    records = clean_records(frame.head(limit))
    return {
        "filename": filename,
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "preview": records,
    }


def preview_drug_candidates_upload(data, filename, limit=12):
    frame = read_uploaded_table(data, filename)
    if frame.empty:
        raise ValueError("약물 후보 파일에 읽을 수 있는 행이 없습니다.")
    records = clean_records(frame)
    return {
        "filename": filename,
        "row_count": int(len(frame)),
        "rows": records,
        "columns": [str(col) for col in frame.columns],
        "preview": records[:limit],
    }


def data_summary():
    cases = read_tsv("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv")
    cbio = read_tsv("outputs/cbioportal_pancan_glioma_covariates.tsv")
    rna_matrix = ROOT / "data/processed/rna_tpm_primary_overlap.tsv"
    rna_files = read_tsv("outputs/gdc_rna_file_map_primary_overlap.tsv")
    wsi_files = read_tsv("outputs/gdc_wsi_file_map_primary_overlap.tsv")
    merged = cases.merge(cbio, on="case_submitter_id", how="left") if not cases.empty and not cbio.empty else cases
    return {
        "patients": int(len(cases)),
        "lgg": int((cases["project_id"] == "TCGA-LGG").sum()) if "project_id" in cases else 0,
        "gbm": int((cases["project_id"] == "TCGA-GBM").sum()) if "project_id" in cases else 0,
        "rna_files": int(len(rna_files)),
        "wsi_files": int(len(wsi_files)),
        "rna_matrix_exists": rna_matrix.exists(),
        "rna_matrix_size_mb": round(rna_matrix.stat().st_size / 1024 / 1024, 1) if rna_matrix.exists() else 0,
        "subtypes": clean_records(
            merged["pancan_subtype"]
            .fillna("Unknown")
            .replace("", "Unknown")
            .value_counts()
            .rename_axis("subtype")
            .reset_index(name="n")
            if "pancan_subtype" in merged
            else pd.DataFrame()
        ),
    }


def model_comparison():
    base = read_tsv("results/stratified_model_comparison.tsv")
    dev50_runs = [
        ("Clinical", "Clinical/covariates", "results/clinical_dev50_baseline"),
        ("RNA", "50 selected RNA genes", "results/rna_dev50_baseline"),
        ("RNA + Clinical", "50 selected RNA genes + clinical/covariates", "results/rna_clinical_dev50_baseline"),
        ("WSI", "WSI handcrafted features", "results/wsi_dev50_baseline"),
        ("WSI + Clinical", "WSI handcrafted features + clinical/covariates", "results/wsi_clinical_dev50_baseline"),
        (
            "WSI + RNA + Clinical",
            "WSI handcrafted features + 50 selected RNA genes + clinical/covariates",
            "results/wsi_rna_clinical_dev50_baseline",
        ),
        (
            "WSI ResNet18 Embedding",
            "ResNet18 patch embeddings",
            "results/wsi_embedding_resnet18_dev50_baseline",
        ),
        (
            "WSI ResNet18 Embedding + RNA + Clinical",
            "ResNet18 patch embeddings + 50 selected RNA genes + clinical/covariates",
            "results/wsi_embedding_resnet18_rna_clinical_dev50_baseline",
        ),
        (
            "WSI UNI Embedding",
            "UNI patch embeddings",
            "results/wsi_embedding_uni_dev50_baseline",
        ),
        (
            "WSI UNI Embedding + Clinical",
            "UNI patch embeddings + clinical/covariates",
            "results/wsi_embedding_uni_clinical_dev50_baseline",
        ),
        (
            "WSI UNI Embedding + RNA",
            "UNI patch embeddings + 50 selected RNA genes",
            "results/wsi_embedding_uni_rna_dev50_baseline",
        ),
        (
            "WSI UNI Embedding + RNA + Clinical",
            "UNI patch embeddings + 50 selected RNA genes + clinical/covariates",
            "results/wsi_embedding_uni_rna_clinical_dev50_baseline",
        ),
        (
            "WSI CONCH Embedding",
            "CONCH patch embeddings",
            "results/wsi_embedding_conch_dev50_baseline",
        ),
        (
            "WSI CONCH Embedding + RNA + Clinical",
            "CONCH patch embeddings + 50 selected RNA genes + clinical/covariates",
            "results/wsi_embedding_conch_rna_clinical_dev50_baseline",
        ),
    ]
    dev_rows = []
    for model_name, input_label, run_dir in dev50_runs:
        metrics = read_key_value(f"{run_dir}/cox_metrics.tsv")
        if not metrics:
            continue
        dev_rows.append(
            {
                "cohort": "WSI dev 50",
                "model": model_name,
                "input": input_label,
                "n_cases": metrics.get("n_cases"),
                "c_index_cv": metrics.get("c_index_cv"),
                "logrank_p": metrics.get("logrank_p"),
                "folds": metrics.get("folds"),
                "n_rna_features_requested": metrics.get("n_rna_features_requested"),
                "n_clinical_features": metrics.get("n_clinical_features"),
                "run_dir": run_dir,
            }
        )
    if dev_rows:
        base = pd.concat([base, pd.DataFrame(dev_rows)], ignore_index=True)
    return clean_records(base)


def top_genes():
    frame = read_tsv("results/rna_interpretability_top_genes.tsv")
    if frame.empty:
        return []
    frame["direction_label"] = frame["risk_direction"].map(
        {
            "higher_expression_higher_risk": "Raises risk",
            "higher_expression_lower_risk": "Lowers risk",
        }
    ).fillna(frame["risk_direction"])
    return clean_records(frame)


def drug_repurposing():
    clue_candidates_with_tau_path = RESULTS / "drug_repurposing_candidates_with_clue_tau.tsv"
    clue_candidates_ready_path = RESULTS / "drug_repurposing_candidates_clue_ready.tsv"
    clue_candidates_path = clue_candidates_with_tau_path if clue_candidates_with_tau_path.exists() else clue_candidates_ready_path
    clue_summary_path = RESULTS / "clue_lincs_reversal_summary.json"
    final_shortlist_path = RESULTS / "drug_repurposing_final_shortlist.tsv"
    final_summary_path = RESULTS / "drug_repurposing_final_summary.json"
    validation_matrix_with_tau_path = RESULTS / "drug_repurposing_validation_matrix_with_clue_tau.tsv"
    validation_matrix_base_path = RESULTS / "drug_repurposing_validation_matrix.tsv"
    validation_matrix_path = validation_matrix_with_tau_path if validation_matrix_with_tau_path.exists() else validation_matrix_base_path
    validation_summary_path = RESULTS / "drug_repurposing_validation_summary.json"
    normal_candidates_path = RESULTS / "drug_repurposing_candidates_gtex_brain.tsv"
    normal_deg_path = RESULTS / "gbm_vs_gtex_brain_deg_targeted.tsv"
    normal_summary_path = RESULTS / "drug_repurposing_gtex_brain_summary.json"
    clue_summary = {}
    if clue_summary_path.exists():
        clue_summary = json.loads(clue_summary_path.read_text(encoding="utf-8"))
    final_summary = {}
    if final_summary_path.exists():
        final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    validation_summary = {}
    if validation_summary_path.exists():
        validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    if clue_candidates_path.exists() and normal_deg_path.exists():
        candidates = read_tsv(clue_candidates_path.relative_to(ROOT))
        deg = read_tsv("results/gbm_vs_gtex_brain_deg_targeted.tsv")
        summary_path = normal_summary_path
    elif normal_candidates_path.exists() and normal_deg_path.exists():
        candidates = read_tsv("results/drug_repurposing_candidates_gtex_brain.tsv")
        deg = read_tsv("results/gbm_vs_gtex_brain_deg_targeted.tsv")
        summary_path = normal_summary_path
    else:
        candidates = read_tsv("results/drug_repurposing_candidates.tsv")
        deg = read_tsv("results/gbm_deg_proxy.tsv")
        summary_path = RESULTS / "drug_repurposing_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    deg_cols = [
        "gene",
        "comparison",
        "direction",
        "log2_fold_change",
        "fdr_bh",
        "gbm_mean_tpm",
        "control_mean_tpm",
        "survival_risk_direction",
        "survival_coef",
    ]
    candidate_cols = [
        "rank",
        "drug_name",
        "repurposing_score",
        "primary_target",
        "targets",
        "target_count",
        "primary_target_direction",
        "primary_target_log2_fc",
        "primary_target_fdr",
        "interaction_types",
        "sources",
        "approved",
        "antineoplastic",
        "immunotherapy",
        "final_repurposing_score",
        "local_reversal_prior",
        "local_reversal_reason",
        "clue_tau_score",
        "clue_connectivity_direction",
        "pert_id",
        "clue_pert_iname",
        "clue_status",
        "comparison",
        "ranking_note",
    ]
    shortlist_cols = [
        "shortlist_rank",
        "candidate_tier",
        "drug_name",
        "primary_target",
        "final_repurposing_score",
        "repurposing_score",
        "primary_target_direction",
        "primary_target_log2_fc",
        "primary_target_fdr",
        "local_reversal_prior",
        "mechanism_summary_ko",
        "approved",
        "antineoplastic",
        "clue_status",
        "clue_tau_score",
        "clue_connectivity_direction",
        "pert_id",
        "clue_pert_iname",
        "validation_status_ko",
        "rationale_ko",
        "next_validation_ko",
    ]
    shortlist = read_tsv("results/drug_repurposing_final_shortlist.tsv") if final_shortlist_path.exists() else pd.DataFrame()
    validation_cols = [
        "validation_rank",
        "validation_tier",
        "drug_name",
        "primary_target",
        "validation_score",
        "final_repurposing_score",
        "bbb_rule_score",
        "bbb_rule_label",
        "molecular_weight",
        "xlogp",
        "tpsa",
        "gbm_trial_count",
        "gbm_trial_nct_ids",
        "pert_id",
        "clue_pert_iname",
        "clue_tau_score",
        "external_lookup_status",
        "validation_comment_ko",
    ]
    validation = read_tsv(validation_matrix_path.relative_to(ROOT)) if validation_matrix_path.exists() else pd.DataFrame()
    if not deg.empty:
        deg = deg.sort_values(["fdr_bh", "abs_log2_fold_change"], ascending=[True, False]).head(30)
    return {
        "summary": summary,
        "clue": clue_summary,
        "final_summary": final_summary,
        "validation_summary": validation_summary,
        "shortlist": clean_records(shortlist[[col for col in shortlist_cols if col in shortlist.columns]]),
        "validation_matrix": clean_records(validation[[col for col in validation_cols if col in validation.columns]]),
        "candidates": clean_records(candidates[[col for col in candidate_cols if col in candidates.columns]]),
        "deg": clean_records(deg[[col for col in deg_cols if col in deg.columns]]),
    }


def km_curve(run_key):
    run_dir = RUNS.get(run_key, RUNS["all_rna_clinical"])
    frame = read_tsv(f"{run_dir}/cox_km_curve.tsv")
    return clean_records(frame)


def predictions(run_key, limit=40):
    run_dir = RUNS.get(run_key, RUNS["all_rna_clinical"])
    frame = read_tsv(f"{run_dir}/cox_predictions.tsv")
    if frame.empty:
        return []
    frame = frame.sort_values("risk_score", ascending=False)
    if limit and limit > 0:
        frame = frame.head(limit)
    return clean_records(frame)


def patient_gene_contrib(cohort_key):
    paths = {
        "all": "results/interpretability_all_rna_clinical_fast/patient_top_gene_contributions.tsv",
        "lgg": "results/interpretability_lgg_rna_clinical_fast/patient_top_gene_contributions.tsv",
        "gbm": "results/interpretability_gbm_rna_clinical_fast/patient_top_gene_contributions.tsv",
    }
    return clean_records(read_tsv(paths.get(cohort_key, paths["all"])))


def wsi_dataset_key(requested_key="dev50"):
    if requested_key in WSI_DATASETS:
        processed_dir = ROOT / WSI_DATASETS[requested_key]["processed_dir"]
        if (processed_dir / "slide_summary.tsv").exists():
            return requested_key
    for key, config in WSI_DATASETS.items():
        processed_dir = ROOT / config["processed_dir"]
        if (processed_dir / "slide_summary.tsv").exists():
            return key
    return requested_key


def wsi_dataset(requested_key="dev50"):
    dataset_key = wsi_dataset_key(requested_key)
    config = WSI_DATASETS.get(dataset_key, WSI_DATASETS["dev50"])
    processed_dir = config["processed_dir"]
    slides = read_tsv(f"{processed_dir}/slide_summary.tsv")
    patches = read_tsv(f"{processed_dir}/patch_images/patch_image_index.tsv")
    patch_importance = read_tsv("results/wsi_uni_patch_importance.tsv")
    patch_heatmaps = read_tsv("results/wsi_uni_patch_heatmaps.tsv")
    if not slides.empty:
        slides["overlay_url"] = slides["overlay_path"].apply(lambda value: f"/wsi-image?path={value}")
        slides["thumbnail_url"] = slides["thumbnail_path"].apply(lambda value: f"/wsi-image?path={value}")
        def risk_overlay_url(row):
            risk_path = ROOT / processed_dir / str(row["file_id"]) / "risk_overlay_uni.jpg"
            return f"/wsi-image?path={risk_path.relative_to(ROOT)}" if risk_path.exists() else row["overlay_url"]

        slides["risk_overlay_url"] = slides.apply(risk_overlay_url, axis=1)
    if not patches.empty:
        patches["patch_image_url"] = patches["patch_image_path"].apply(lambda value: f"/wsi-image?path={value}")
        if not patch_importance.empty and {"file_id", "rank"}.issubset(patch_importance.columns):
            patches["rank"] = pd.to_numeric(patches["rank"], errors="coerce").astype("Int64")
            patch_importance["rank"] = pd.to_numeric(patch_importance["rank"], errors="coerce").astype("Int64")
            keep_cols = [
                "file_id",
                "rank",
                "patch_risk_score",
                "patch_importance_abs",
                "patch_risk_direction",
                "patch_risk_rank",
                "patch_importance_rank",
                "patch_importance_norm",
            ]
            patches = patches.merge(
                patch_importance[[col for col in keep_cols if col in patch_importance.columns]],
                on=["file_id", "rank"],
                how="left",
            )
        if not patch_heatmaps.empty and {"file_id", "rank", "patch_attention_heatmap_path"}.issubset(patch_heatmaps.columns):
            patches["rank"] = pd.to_numeric(patches["rank"], errors="coerce").astype("Int64")
            patch_heatmaps["rank"] = pd.to_numeric(patch_heatmaps["rank"], errors="coerce").astype("Int64")
            patch_heatmaps = patch_heatmaps[["file_id", "rank", "patch_attention_heatmap_path"]].copy()
            patches = patches.merge(patch_heatmaps, on=["file_id", "rank"], how="left")
            patches["patch_attention_heatmap_url"] = patches["patch_attention_heatmap_path"].apply(
                lambda value: f"/wsi-image?path={value}" if isinstance(value, str) and value else None
            )
    return {
        "dataset": dataset_key,
        "label": config["label"],
        "slides": clean_records(slides),
        "patches": clean_records(patches),
    }


def wsi_features(requested_key="dev50"):
    dataset_key = wsi_dataset_key(requested_key)
    config = WSI_DATASETS.get(dataset_key, WSI_DATASETS["dev50"])
    frame = read_tsv(f"{config['processed_dir']}/wsi_handcrafted_features.tsv")
    if frame.empty:
        return []
    display_cols = [
        "case_submitter_id",
        "project_id",
        "file_id",
        "wsi_patches_used",
        "wsi_tissue_fraction_thumbnail",
        "wsi_tissue_area_megapixels",
        "wsi_brightness_mean_mean",
        "wsi_saturation_mean_mean",
        "wsi_dark_pixel_ratio_mean",
        "wsi_purple_pixel_ratio_mean",
        "wsi_pink_pixel_ratio_mean",
    ]
    return clean_records(frame[[col for col in display_cols if col in frame.columns]])


def run_rna_clinical_prediction(rna_file, clinical_json, out_dir):
    if not (MODEL_DIR / "model.joblib").exists():
        raise FileNotFoundError("Final RNA+Clinical model is missing. Run the final model training task first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/predict_survival_risk.py"),
        "--model-dir",
        str(MODEL_DIR),
        "--rna-file",
        str(rna_file),
        "--clinical-json",
        str(clinical_json),
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Prediction failed."
        raise RuntimeError(message)

    with (out_dir / "prediction.json").open() as handle:
        prediction = json.load(handle)
    gene_contrib = read_tsv(out_dir.relative_to(ROOT) / "top_gene_contributions.tsv")
    clinical_contrib = read_tsv(out_dir.relative_to(ROOT) / "top_clinical_contributions.tsv")
    return {
        "prediction": prediction,
        "gene_contributions": clean_records(gene_contrib),
        "clinical_contributions": clean_records(clinical_contrib),
    }


def example_prediction():
    return run_rna_clinical_prediction(
        MODEL_DIR / "example_inputs/example_rna.tsv",
        MODEL_DIR / "example_inputs/example_clinical.json",
        OUTPUTS / "predictions/dashboard_example",
    )


HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TCGA 교종 생존 예측 대시보드</title>
  <style>
    :root {
      --bg: #f5f6f7;
      --panel: #ffffff;
      --ink: #1a1c21;
      --muted: #737880;
      --line: #e0e3e5;
      --soft-line: #edf0f4;
      --blue: #2663eb;
      --red: #c44536;
      --green: #257a52;
      --amber: #ad741f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      background: var(--bg);
    }
    .wrap {
      width: min(1360px, calc(100% - 80px));
      margin: 0 auto;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 32px 0 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 800;
    }
    .subtitle {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      font-size: 12px;
      color: var(--muted);
    }
    .header-actions {
      display: flex;
      align-items: flex-end;
      gap: 12px;
      flex-direction: column;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      background: #ffffff;
      white-space: nowrap;
    }
    main {
      padding: 0 0 36px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    .tab-button {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 20px;
      background: #ffffff;
      color: var(--ink);
      font-size: 14px;
      font-weight: 650;
      white-space: nowrap;
    }
    .tab-button.active {
      border-color: var(--blue);
      background: var(--blue);
      color: #ffffff;
      box-shadow: 0 8px 18px rgba(38, 99, 235, 0.18);
    }
    .tab-panel {
      display: none;
    }
    .tab-panel.active {
      display: block;
    }
    .analysis-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .analysis-tab-button {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 14px;
      background: #ffffff;
      color: var(--ink);
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }
    .analysis-tab-button.active {
      border-color: #1a1c21;
      background: #1a1c21;
      color: #ffffff;
    }
    .analysis-panel {
      display: none;
    }
    .analysis-panel.active {
      display: block;
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .stats {
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }
    .two {
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
      margin-top: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-width: 0;
      box-shadow: 0 1px 1px rgba(20, 23, 31, 0.02);
    }
    .metric .label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    .metric .value {
      margin-top: 6px;
      font-size: 28px;
      font-weight: 800;
      line-height: 1.2;
    }
    .metric .note {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      line-height: 1.35;
    }
    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    label {
      font-size: 13px;
      color: var(--muted);
    }
    select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #ffffff;
      min-width: 220px;
      color: var(--ink);
    }
    .lang-select {
      min-width: 128px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--soft-line);
      padding: 8px 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fafafa;
    }
    .table-scroll {
      overflow: auto;
      max-height: 420px;
    }
    .table-scroll thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      box-shadow: inset 0 -1px 0 var(--soft-line);
    }
    .wsi-table {
      table-layout: fixed;
      min-width: 920px;
    }
    .wsi-table th,
    .wsi-table td {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .wsi-table .col-patient { width: 170px; }
    .wsi-table .col-project { width: 130px; }
    .wsi-table .col-patches { width: 96px; }
    .wsi-table .col-ratio { width: 116px; }
    .wsi-table .col-status { width: 110px; }
    .wsi-row {
      cursor: pointer;
    }
    .wsi-row:hover td {
      background: #f7faff;
    }
    .wsi-row.is-selected td {
      background: #eef5ff;
      box-shadow: inset 3px 0 0 var(--blue);
    }
    .wsi-selected-card {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }
    .wsi-selected-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .wsi-selected-title {
      font-weight: 800;
      font-size: 15px;
    }
    .wsi-selected-subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
      line-height: 1.35;
      word-break: break-all;
    }
    .wsi-detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .wsi-detail-item {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      padding: 8px;
      background: #ffffff;
      min-width: 0;
    }
    .wsi-detail-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .wsi-detail-value {
      margin-top: 4px;
      font-size: 16px;
      font-weight: 800;
      line-height: 1.25;
    }
    .performance-workbench {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 16px;
      align-items: stretch;
    }
    .performance-side {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      min-width: 0;
    }
    .performance-side h3 {
      margin: 0 0 10px;
      font-size: 13px;
      line-height: 1.35;
    }
    .performance-cells {
      display: grid;
      gap: 7px;
    }
    .performance-cell {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      background: #ffffff;
      padding: 9px 10px;
      text-align: left;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
      font-size: 12px;
    }
    .performance-cell:hover {
      border-color: #b9c7da;
      background: #f7faff;
    }
    .performance-cell.is-selected {
      border-color: var(--blue);
      background: #eef5ff;
      box-shadow: inset 3px 0 0 var(--blue);
    }
    .performance-cell-name {
      font-weight: 750;
      display: block;
      line-height: 1.25;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .performance-cell-meta {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
      line-height: 1.25;
    }
    .performance-cell-value {
      font-weight: 850;
      font-size: 14px;
    }
    .patch-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .patch-tile {
      padding: 0;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #f9fafb;
      position: relative;
      aspect-ratio: 1 / 1;
      display: block;
      cursor: pointer;
    }
    .patch-tile.raises-risk {
      border-color: rgba(196, 69, 54, 0.75);
    }
    .patch-tile.lowers-risk {
      border-color: rgba(38, 99, 235, 0.65);
    }
    .patch-tile:hover,
    .patch-tile.is-selected {
      border-color: var(--blue);
      box-shadow: 0 0 0 2px rgba(38, 99, 235, 0.16);
    }
    .patch-tile.raises-risk.is-selected,
    .patch-tile.raises-risk:hover {
      border-color: var(--red);
      box-shadow: 0 0 0 2px rgba(196, 69, 54, 0.2);
    }
    .patch-tile img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .patch-badge {
      position: absolute;
      left: 6px;
      bottom: 6px;
      border-radius: 999px;
      padding: 2px 6px;
      background: rgba(26, 28, 33, 0.72);
      color: #ffffff;
      font-size: 10px;
      font-weight: 750;
    }
    .patch-meta {
      margin-top: 8px;
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .panel-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .panel-title-row h2 {
      margin: 0;
    }
    .small-button {
      padding: 7px 10px;
      font-size: 12px;
      white-space: nowrap;
    }
    .chart {
      width: 100%;
      min-height: 260px;
    }
    .legend {
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 5px;
      border-radius: 2px;
      vertical-align: -1px;
    }
    .section {
      margin-top: 16px;
    }
    .upload-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
    }
    .file-control {
      display: grid;
      gap: 6px;
    }
    .drop-control {
      border: 1px dashed #b9c7da;
      border-radius: 8px;
      padding: 14px;
      background: #fbfcff;
    }
    .drop-control:hover {
      border-color: var(--blue);
      background: #f5f8ff;
    }
    input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fafafa;
      color: var(--ink);
    }
    input[type="text"],
    input[type="number"],
    .clinical-field select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
    }
    .clinical-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .clinical-field {
      display: grid;
      gap: 5px;
    }
    .clinical-field label,
    .file-control label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .rna-preview {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }
    .rna-preview-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .rna-preview table {
      margin: 0;
    }
    .patient-context {
      border: 1px solid #d9e5ff;
      border-radius: 8px;
      background: #fbfdff;
      padding: 13px;
      margin-bottom: 14px;
    }
    .patient-context-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
    }
    .patient-context-title {
      font-size: 16px;
      font-weight: 850;
    }
    .patient-context-subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
      line-height: 1.4;
    }
    .patient-context-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .patient-context-metric {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      background: #ffffff;
      padding: 8px;
      min-width: 0;
    }
    .patient-context-metric div:first-child {
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      margin-bottom: 4px;
    }
    .patient-context-metric div:last-child {
      font-size: 15px;
      font-weight: 850;
      overflow-wrap: anywhere;
    }
    .patient-search-row {
      display: grid;
      grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }
    .patient-list {
      display: grid;
      gap: 6px;
      max-height: 220px;
      overflow: auto;
    }
    .patient-list button {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
      align-items: center;
      text-align: left;
      font-size: 12px;
    }
    .patient-list button.is-selected {
      border-color: var(--blue);
      background: #eef5ff;
      box-shadow: inset 3px 0 0 var(--blue);
    }
    .wsi-upload-preview {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .uploaded-wsi-preview {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }
    .uploaded-wsi-preview img {
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      display: block;
      background: #f8fafc;
    }
    .patient-drug-cards {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .patient-drug-card {
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
      min-width: 0;
    }
    .patient-drug-card-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 850;
      margin-bottom: 6px;
    }
    .patient-drug-card-title span {
      color: var(--blue);
      font-size: 12px;
      white-space: nowrap;
    }
    .patient-drug-card p {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .patient-drug-match {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }
    .patient-drug-match span {
      border: 1px solid rgba(38, 99, 235, 0.22);
      border-radius: 999px;
      background: #eef5ff;
      color: var(--blue);
      padding: 3px 7px;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .patient-drug-scoreline {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      margin-top: 8px;
    }
    .patient-drug-scoreline div {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      background: #fbfcfd;
      padding: 6px;
      min-width: 0;
    }
    .patient-drug-scoreline span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      margin-bottom: 2px;
    }
    .patient-drug-scoreline strong {
      display: block;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      background: #ffffff;
      color: var(--ink);
      cursor: pointer;
      font-weight: 650;
    }
    button.primary {
      border-color: var(--blue);
      background: var(--blue);
      color: #ffffff;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fafafa;
      margin-bottom: 12px;
    }
    .risk-high { color: var(--red); }
    .risk-low { color: var(--blue); }
    .gene-row {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr) 72px;
      align-items: center;
      gap: 10px;
      margin: 8px 0;
      font-size: 13px;
    }
    .shortlist-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .shortlist-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 13px;
      min-width: 0;
    }
    .shortlist-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
      margin-bottom: 8px;
    }
    .shortlist-name {
      font-size: 15px;
      font-weight: 850;
      overflow-wrap: anywhere;
    }
    .shortlist-score {
      font-size: 13px;
      font-weight: 800;
      color: var(--blue);
    }
    .shortlist-target {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      margin-bottom: 8px;
    }
    .shortlist-rationale {
      color: var(--ink);
      font-size: 12px;
      line-height: 1.5;
      margin: 0 0 9px;
    }
    .shortlist-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .shortlist-tag {
      border: 1px solid var(--soft-line);
      border-radius: 999px;
      background: #ffffff;
      color: var(--muted);
      padding: 3px 7px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .shortlist-tag.primary {
      border-color: rgba(38, 99, 235, 0.25);
      color: var(--blue);
      background: #eef5ff;
    }
    .risk-view-panel,
    .drug-view-panel {
      display: none;
      gap: 16px;
    }
    .risk-view-panel.active,
    .drug-view-panel.active {
      display: grid;
    }
    #risk-view-prediction,
    #risk-view-patientDrugs,
    #drug-view-overview {
      grid-template-columns: 1fr;
    }
    #drug-view-table {
      grid-template-columns: 1fr;
    }
    #drug-view-register {
      grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
      align-items: start;
    }
    .drug-register-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .drug-register-grid .wide {
      grid-column: 1 / -1;
    }
    .drug-register-grid textarea {
      min-height: 88px;
      resize: vertical;
    }
    .drug-upload-zone {
      border: 1px dashed #b8c8e8;
      border-radius: 8px;
      background: #fbfdff;
      padding: 14px;
      margin-top: 12px;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .drug-upload-zone.drag-over {
      border-color: var(--blue);
      background: #eef5ff;
    }
    .drug-user-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      max-height: 360px;
      overflow: auto;
    }
    .drug-user-row {
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
    }
    .drug-user-row strong {
      display: block;
      margin-bottom: 4px;
    }
    .drug-user-row span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .drug-user-row button {
      margin-top: 8px;
      padding: 5px 8px;
      font-size: 11px;
    }
    .drug-rank-bars {
      display: grid;
      gap: 12px;
      max-width: 980px;
      margin-top: 8px;
    }
    .drug-rank-row {
      display: grid;
      grid-template-columns: minmax(220px, 310px) minmax(0, 1fr) 56px;
      gap: 12px;
      align-items: start;
      font-size: 13px;
    }
    .drug-rank-name {
      font-weight: 750;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .drug-rank-track {
      height: 12px;
      background: #edf1f4;
      border-radius: 3px;
      overflow: hidden;
      margin-top: 3px;
    }
    .drug-rank-fill {
      display: block;
      height: 100%;
      background: #c84f42;
    }
    .drug-rank-score {
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
      text-align: right;
    }
    .drug-table-note {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      background: #f7faff;
      border: 1px solid #d9e5ff;
      border-radius: 6px;
      padding: 7px 9px;
      margin-bottom: 10px;
      font-size: 12px;
      line-height: 1.35;
    }
    .drug-validation-table {
      min-width: 1120px;
      font-size: 12px;
    }
    .drug-validation-table th,
    .drug-validation-table td {
      padding: 7px 8px;
    }
    .drug-row {
      cursor: pointer;
    }
    .drug-row:hover td {
      background: #f7faff;
    }
    .drug-row.is-selected td {
      background: #eef5ff;
      box-shadow: inset 3px 0 0 var(--blue);
    }
    .score-chip {
      display: inline-flex;
      min-width: 54px;
      justify-content: center;
      border-radius: 5px;
      padding: 3px 7px;
      background: #eef2f6;
      color: var(--ink);
      font-weight: 800;
    }
    .tau-chip {
      display: inline-flex;
      min-width: 58px;
      justify-content: center;
      border-radius: 5px;
      padding: 3px 7px;
      font-weight: 800;
      background: #eef8f3;
      color: var(--green);
    }
    .tau-chip.mimicry {
      background: #fff2f0;
      color: var(--red);
    }
    .drug-selected-panel {
      border: 1px solid var(--blue);
      border-radius: 8px;
      background: #fbfdff;
      padding: 15px;
    }
    .drug-selected-title {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 17px;
      font-weight: 850;
      margin-bottom: 7px;
    }
    .drug-selected-title span {
      color: var(--blue);
      font-size: 12px;
      background: #eef5ff;
      border: 1px solid #d8e5ff;
      border-radius: 999px;
      padding: 3px 8px;
    }
    .drug-selected-comment {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      margin: 0 0 12px;
    }
    .drug-selected-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }
    .drug-selected-metric {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      background: #ffffff;
      padding: 9px;
      min-width: 0;
    }
    .drug-selected-metric div:first-child {
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      margin-bottom: 4px;
    }
    .drug-selected-metric div:last-child {
      font-size: 15px;
      font-weight: 850;
      overflow-wrap: anywhere;
    }
    .drug-reason-list {
      display: grid;
      gap: 7px;
      margin: 10px 0 12px;
    }
    .drug-reason-item {
      border: 1px solid var(--soft-line);
      border-radius: 6px;
      background: #ffffff;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
    }
    .drug-reason-item strong {
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
    }
    .compact-reference {
      display: none;
    }
    .bar-track {
      position: relative;
      height: 18px;
      background: #eef2f6;
      border-radius: 4px;
      overflow: hidden;
    }
    .bar-zero {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      border-left: 1px solid #56616f;
    }
    .bar {
      position: absolute;
      top: 0;
      bottom: 0;
    }
    .bar.pos { left: 50%; background: var(--red); }
    .bar.neg { right: 50%; background: var(--blue); }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .footer-note {
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
    }
    .image-frame {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #f9fafb;
      aspect-ratio: 16 / 7;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .image-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .patch-preview-frame {
      aspect-ratio: 4 / 3;
    }
    .patch-preview-pair {
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 8px;
      box-sizing: border-box;
    }
    .patch-preview-item {
      min-width: 0;
      height: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .patch-preview-item img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .patch-preview-label {
      position: absolute;
      left: 6px;
      top: 6px;
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(26, 28, 33, 0.74);
      color: #ffffff;
      font-size: 10px;
      font-weight: 750;
    }
    .image-modal {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(15, 23, 42, 0.72);
    }
    .image-modal.active {
      display: flex;
    }
    .image-modal-card {
      width: min(1280px, 96vw);
      max-height: 92vh;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 24px 80px rgba(15, 23, 42, 0.34);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
    }
    .image-modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--soft-line);
    }
    .image-modal-title {
      font-weight: 800;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .image-modal-body {
      min-height: 0;
      overflow: auto;
      padding: 12px;
      background: #f5f7fa;
    }
    .image-modal-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }
    .image-modal-figure {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      padding: 10px;
    }
    .image-modal-figure img {
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
      display: block;
    }
    .image-modal-caption {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .image-modal-foot {
      padding: 10px 14px;
      border-top: 1px solid var(--soft-line);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 900px) {
      .wrap { width: min(100% - 32px, 1360px); }
      .topbar { align-items: flex-start; flex-direction: column; }
      .header-actions { align-items: flex-start; }
      .status { justify-content: flex-start; }
      .stats, .two, .upload-grid, .clinical-grid, .patient-context-grid, .patient-search-row, .patient-drug-cards, .shortlist-grid, .drug-selected-grid, #drug-view-register, .drug-register-grid { grid-template-columns: 1fr; }
      .drug-register-grid .wide { grid-column: auto; }
      .drug-rank-row { grid-template-columns: minmax(0, 1fr) 48px; }
      .drug-rank-name { grid-column: 1 / -1; }
      .performance-workbench { grid-template-columns: 1fr; }
      .patch-preview-pair { grid-template-columns: 1fr; }
      .image-modal-grid { grid-template-columns: 1fr; }
      .tabs { margin-bottom: 16px; }
      .wsi-detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1 data-i18n="appTitle">TCGA 교종 멀티모달 생존 예측 대시보드</h1>
        <p class="subtitle" data-i18n="appSubtitle">WSI 통합 전 RNA와 임상정보 기반 생존분석 결과</p>
      </div>
      <div class="header-actions">
        <div class="controls" style="margin-bottom:0">
          <label for="languageSelect" data-i18n="language">언어</label>
          <select id="languageSelect" class="lang-select">
            <option value="ko">한국어</option>
            <option value="en">English</option>
          </select>
        </div>
        <div class="status" id="statusPills"></div>
      </div>
    </div>
  </header>
  <main class="wrap">
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab-button active" type="button" data-tab-target="overview" data-i18n="riskPredictionTab">위험 예측</button>
      <button class="tab-button" type="button" data-tab-target="performance" data-i18n="performanceTab">모델 성능 & 생존분석</button>
      <button class="tab-button" type="button" data-tab-target="wsi" data-i18n="wsiTab">WSI 병리</button>
      <button class="tab-button" type="button" data-tab-target="drugs" data-i18n="drugCandidateTab">약물 후보</button>
    </nav>

    <div class="tab-panel active" id="tab-overview">
      <section class="grid stats" id="stats"></section>

      <div class="analysis-tabs section" aria-label="Risk prediction views">
        <button class="analysis-tab-button active" type="button" data-risk-view-target="prediction" data-i18n="riskInputView">위험 예측 실행</button>
        <button class="analysis-tab-button" type="button" data-risk-view-target="patientDrugs" data-i18n="patientDrugView">환자별 약물 후보</button>
      </div>

      <section class="risk-view-panel active" id="risk-view-prediction">
        <div class="grid two">
          <div class="panel">
            <h2 data-i18n="inferencePrototype">RNA + 임상정보 예측 엔진</h2>
            <form id="predictionForm">
              <div class="upload-grid">
                <div class="file-control drop-control">
                  <label for="rnaFile" data-i18n="rnaTableUpload">RNA 표 파일</label>
                  <input id="rnaFile" name="rna_file" type="file" accept=".tsv,.txt,.csv,.xlsx,.xls" />
                  <div class="hint" data-i18n="rnaUploadHint">gene/tpm 컬럼이 있는 TSV 또는 CSV를 드래그하면 아래 셀에 미리 표시됩니다.</div>
                </div>
                <div class="file-control drop-control">
                  <label for="wsiImageFile" data-i18n="wsiImageUpload">WSI 병리 이미지</label>
                  <input id="wsiImageFile" name="wsi_image_file" type="file" accept="image/*,.svs,.tif,.tiff" />
                  <div class="hint" data-i18n="wsiUploadHint">이번 단계에서는 환자 케이스에 이미지를 함께 등록하고, 실제 WSI 기반 예측은 WSI embedding 모델 학습 후 반영합니다.</div>
                  <div class="wsi-upload-preview" id="wsiUploadPreview" data-i18n="wsiUploadEmpty">선택된 WSI 이미지가 없습니다.</div>
                </div>
              </div>
              <div class="rna-preview" id="rnaPreviewBox">
                <div class="rna-preview-head">
                  <span data-i18n="rnaCellPreview">RNA 셀 미리보기</span>
                  <span id="rnaPreviewMeta" data-i18n="previewEmpty">파일을 선택하면 미리보기가 표시됩니다.</span>
                </div>
                <div class="table-scroll">
                  <table id="rnaPreviewTable"></table>
                </div>
              </div>
              <h3 class="section" data-i18n="clinicalCells">임상정보 입력 셀</h3>
              <div class="clinical-grid">
                <div class="clinical-field">
                  <label for="caseSubmitterId" data-i18n="caseId">환자 ID</label>
                  <input id="caseSubmitterId" type="text" value="TCGA-06-5412" />
                </div>
                <div class="clinical-field">
                  <label for="projectId" data-i18n="project">프로젝트</label>
                  <select id="projectId">
                    <option value="TCGA-GBM">TCGA-GBM</option>
                    <option value="TCGA-LGG">TCGA-LGG</option>
                  </select>
                </div>
                <div class="clinical-field">
                  <label for="gender" data-i18n="sex">성별</label>
                  <select id="gender">
                    <option value="Unknown">Unknown</option>
                    <option value="male">male</option>
                    <option value="female">female</option>
                  </select>
                </div>
                <div class="clinical-field">
                  <label for="ageAtDiagnosis" data-i18n="age">진단 시 나이</label>
                  <input id="ageAtDiagnosis" type="number" min="0" max="120" step="0.1" value="78.7" />
                </div>
                <div class="clinical-field">
                  <label for="tumorGrade" data-i18n="tumorGrade">종양 등급</label>
                  <input id="tumorGrade" type="text" value="Unknown" />
                </div>
                <div class="clinical-field">
                  <label for="primaryDiagnosis" data-i18n="primaryDiagnosis">진단명</label>
                  <input id="primaryDiagnosis" type="text" value="Glioblastoma" />
                </div>
                <div class="clinical-field">
                  <label for="pancanSubtype" data-i18n="subtype">분자아형</label>
                  <input id="pancanSubtype" type="text" value="GBM_IDHwt" />
                </div>
                <div class="clinical-field">
                  <label for="cancerTypeDetailed" data-i18n="cancerTypeDetailed">상세 암종</label>
                  <input id="cancerTypeDetailed" type="text" value="Glioblastoma Multiforme" />
                </div>
                <div class="clinical-field">
                  <label for="tumorType" data-i18n="tumorType">종양 유형</label>
                  <input id="tumorType" type="text" value="Glioblastoma Multiforme (GBM), Untreated" />
                </div>
              </div>
              <div class="controls" style="margin-top:12px; margin-bottom:0">
                <button class="primary" type="submit" data-i18n="predict">예측 실행</button>
                <button type="button" id="examplePredictButton" data-i18n="useExample">예시 실행</button>
                <span class="hint" id="predictionStatus"></span>
              </div>
            </form>
          </div>
          <div class="panel">
            <h2 data-i18n="predictionResult">예측 결과</h2>
            <div id="predictionResult" class="hint" data-i18n="predictionEmpty">아직 실행된 예측이 없습니다.</div>
          </div>
        </div>
      </section>

      <section class="risk-view-panel" id="risk-view-patientDrugs">
        <div class="panel">
          <div class="patient-search-row">
            <div class="clinical-field">
              <label for="drugPatientSearch" data-i18n="registeredPatientSearch">등록 환자 검색</label>
              <input id="drugPatientSearch" type="text" placeholder="TCGA-06-5412" />
            </div>
            <div id="drugPatientList" class="patient-list"></div>
          </div>
          <div class="patient-context" id="drugPatientContext"></div>
        </div>
      </section>
    </div>

    <div class="tab-panel" id="tab-drugs">
      <div class="analysis-tabs section" aria-label="Drug repurposing views">
        <button class="analysis-tab-button active" type="button" data-drug-view-target="register" data-i18n="drugRegisterView">후보 약물 등록</button>
        <button class="analysis-tab-button" type="button" data-drug-view-target="table" data-i18n="drugTableView">통합 후보 테이블</button>
        <button class="analysis-tab-button" type="button" data-drug-view-target="overview" data-i18n="drugOverviewView">후보 순위 개요</button>
      </div>

      <section class="drug-view-panel active" id="drug-view-register">
        <div class="panel">
          <div class="panel-title-row">
            <div>
              <h2 data-i18n="drugRegisterTitle">후보 약물 직접 등록</h2>
              <p class="hint" data-i18n="drugRegisterHint">약물명과 타깃 유전자를 입력하면 통합 후보 테이블과 환자별 약물 후보 계산에 바로 반영됩니다.</p>
            </div>
          </div>
          <form id="drugRegistrationForm">
            <div class="drug-register-grid">
              <div class="clinical-field">
                <label for="candidateDrugName" data-i18n="drugName">약물</label>
                <input id="candidateDrugName" type="text" placeholder="ExampleDrug-A" />
              </div>
              <div class="clinical-field">
                <label for="candidatePrimaryTarget" data-i18n="target">타깃</label>
                <input id="candidatePrimaryTarget" type="text" placeholder="ABCC3" />
              </div>
              <div class="clinical-field">
                <label for="candidateTargets" data-i18n="targets">타깃 목록</label>
                <input id="candidateTargets" type="text" placeholder="ABCC3;TOP2A" />
              </div>
              <div class="clinical-field">
                <label for="candidateAction" data-i18n="drugAction">작용 방향</label>
                <select id="candidateAction">
                  <option value="inhibitor">inhibitor</option>
                  <option value="activator">activator</option>
                  <option value="unknown">unknown</option>
                </select>
              </div>
              <div class="clinical-field">
                <label for="candidateValidationScore" data-i18n="validationScore">검증 점수</label>
                <input id="candidateValidationScore" type="number" min="0" max="1" step="0.001" value="0.500" />
              </div>
              <div class="clinical-field">
                <label for="candidateClueTau" data-i18n="clueTau">CLUE tau</label>
                <input id="candidateClueTau" type="number" step="0.01" placeholder="-80" />
              </div>
              <div class="clinical-field">
                <label for="candidateBbbLabel" data-i18n="bbbEstimate">BBB 추정</label>
                <select id="candidateBbbLabel">
                  <option value="미확인">미확인</option>
                  <option value="BBB 물성 우호적">BBB 물성 우호적</option>
                  <option value="BBB 물성 경계">BBB 물성 경계</option>
                  <option value="BBB 물성 불리">BBB 물성 불리</option>
                </select>
              </div>
              <div class="clinical-field">
                <label for="candidateApproved" data-i18n="approved">승인</label>
                <select id="candidateApproved">
                  <option value="false">아니오</option>
                  <option value="true">예</option>
                </select>
              </div>
              <div class="clinical-field">
                <label for="candidateAntineoplastic" data-i18n="antineoplastic">항암제</label>
                <select id="candidateAntineoplastic">
                  <option value="false">아니오</option>
                  <option value="true">예</option>
                </select>
              </div>
              <div class="clinical-field">
                <label for="candidateTrialCount" data-i18n="gbmTrials">GBM trial</label>
                <input id="candidateTrialCount" type="number" min="0" step="1" value="0" />
              </div>
              <div class="clinical-field wide">
                <label for="candidateEvidence" data-i18n="evidenceComment">근거 코멘트</label>
                <textarea id="candidateEvidence" placeholder="환자 위험 유전자와 타깃이 겹치며, 추가 검증이 필요한 사용자 등록 후보입니다."></textarea>
              </div>
            </div>
            <div class="controls" style="margin-top:12px; margin-bottom:0">
              <button class="primary" type="submit" data-i18n="addDrugCandidate">후보 추가</button>
              <button type="button" id="drugExampleButton" data-i18n="fillDrugExample">예시 채우기</button>
              <span class="hint" id="drugRegisterStatus"></span>
            </div>
          </form>

          <div class="drug-upload-zone" id="drugUploadZone">
            <div class="file-control drop-control">
              <label for="userDrugFile" data-i18n="drugExcelUpload">엑셀/CSV 후보 파일</label>
              <input id="userDrugFile" type="file" accept=".xlsx,.xls,.csv,.tsv,.txt" />
              <div class="hint" data-i18n="drugUploadHint">drug_name, primary_target, targets, validation_score 같은 컬럼이 있으면 등록 후보로 바로 반영할 수 있습니다.</div>
            </div>
            <div class="rna-preview" id="drugPreviewBox">
              <div class="rna-preview-head">
                <span data-i18n="drugPreviewTitle">후보 파일 미리보기</span>
                <span id="drugPreviewMeta" data-i18n="previewEmpty">파일을 선택하면 미리보기가 표시됩니다.</span>
              </div>
              <div class="table-scroll">
                <table id="drugPreviewTable"></table>
              </div>
            </div>
            <div class="controls" style="margin-top:12px; margin-bottom:0">
              <button type="button" id="applyDrugPreviewButton" data-i18n="applyDrugPreview">미리보기 후보 반영</button>
              <span class="hint" id="drugPreviewStatus"></span>
            </div>
          </div>
        </div>

        <div class="panel">
          <h2 data-i18n="registeredDrugCandidates">등록된 후보 약물</h2>
          <p class="hint" data-i18n="registeredDrugHint">사용자 등록 후보는 브라우저에 저장되고, 기존 후보와 함께 개인화 약물 후보 계산에 사용됩니다.</p>
          <div id="userDrugList" class="drug-user-list"></div>
        </div>
      </section>

      <section class="drug-view-panel" id="drug-view-table">
        <div class="drug-selected-panel" id="drugSelectedDetail"></div>
        <div class="panel">
          <div class="panel-title-row">
            <div>
              <h2 data-i18n="integratedCandidateTable">통합 후보 테이블</h2>
              <p class="hint" id="validationSummary"></p>
            </div>
          </div>
          <div class="drug-table-note" id="clueSummary"></div>
          <div class="table-scroll">
            <table id="drugValidationTable"></table>
          </div>
        </div>
        <section class="panel compact-reference" id="drugRawTableSection">
          <h2 data-i18n="drugCandidateTable">후보 약물 상세표</h2>
          <div class="table-scroll">
            <table id="drugCandidateTable"></table>
          </div>
        </section>
      </section>

      <section class="drug-view-panel" id="drug-view-overview">
        <div class="panel">
          <div class="panel-title-row">
            <div>
              <h2 data-i18n="drugRankOverview">재창출 후보 순위</h2>
              <p class="hint" id="finalDrugSummary"></p>
            </div>
          </div>
          <div class="drug-rank-bars" id="drugCandidateBars"></div>
        </div>

        <div class="panel">
          <h2 data-i18n="coreTargetGenes">핵심 타깃군 - GBM 차등발현 유전자</h2>
          <p class="hint" id="drugSummary"></p>
          <div class="table-scroll">
            <table id="degTable"></table>
          </div>
        </div>
      </section>
    </div>

    <div class="tab-panel" id="tab-performance">
      <section class="panel">
        <div class="analysis-tabs" aria-label="Performance views">
          <button class="analysis-tab-button active" type="button" data-analysis-target="model" data-i18n="modelGraphTab">모델 성능 그래프</button>
          <button class="analysis-tab-button" type="button" data-analysis-target="km" data-i18n="kmGraphTab">Kaplan-Meier 생존곡선</button>
          <button class="analysis-tab-button" type="button" data-analysis-target="genes" data-i18n="geneGraphTab">주요 RNA 위험 유전자</button>
        </div>

        <div class="analysis-panel active" id="analysis-model">
          <div class="controls">
            <h2 style="margin-right:auto" data-i18n="modelPerformance">모델 성능 비교</h2>
            <label for="performanceMode" data-i18n="viewMode">보기</label>
            <select id="performanceMode">
              <option value="cohort" data-i18n="cohortPerformance">코호트별 모델 성능</option>
              <option value="wsi" data-i18n="wsiEmbeddingComparison">WSI 임베딩 비교</option>
            </select>
          </div>
          <div class="performance-workbench">
            <div id="performanceChart" class="chart compact-chart"></div>
            <aside class="performance-side">
              <h3 id="performanceCellTitle" data-i18n="highlightModel">강조할 모델</h3>
              <div id="performanceCells" class="performance-cells"></div>
            </aside>
          </div>
        </div>

        <div class="analysis-panel" id="analysis-km">
          <div class="controls">
            <h2 style="margin-right:auto" data-i18n="kmCurve">Kaplan-Meier 생존곡선</h2>
            <label for="kmRun" data-i18n="run">실험</label>
            <select id="kmRun"></select>
          </div>
          <div id="kmChart" class="chart compact-chart"></div>
        </div>

        <div class="analysis-panel" id="analysis-genes">
          <div class="controls">
            <h2 style="margin-right:auto" data-i18n="topGenes">주요 RNA 위험 유전자</h2>
            <label for="geneCohort" data-i18n="cohort">코호트</label>
            <select id="geneCohort"></select>
            <label for="geneAnalysis" data-i18n="analysis">분석</label>
            <select id="geneAnalysis"></select>
          </div>
          <div id="geneBars"></div>
        </div>
      </section>

      <section class="section">
        <div class="analysis-panel active" id="analysis-model-detail">
          <div class="panel">
            <h2 data-i18n="coreTakeaway">핵심 해석</h2>
            <p class="hint" id="takeaway"></p>
            <div class="table-scroll">
              <table id="comparisonTable"></table>
            </div>
          </div>
        </div>

        <div class="analysis-panel" id="analysis-km-detail">
          <div class="panel">
            <h2 style="margin-right:auto" data-i18n="highestRiskPatients">고위험 예측 환자</h2>
            <div class="controls">
              <label for="predRun" data-i18n="run">실험</label>
              <select id="predRun"></select>
            </div>
            <div class="table-scroll">
              <table id="predictionTable"></table>
            </div>
          </div>
        </div>

        <div class="analysis-panel" id="analysis-genes-detail">
          <div class="panel">
            <h2 style="margin-right:auto" data-i18n="patientContrib">환자별 유전자 기여도</h2>
            <div class="controls">
              <label for="contribCohort" data-i18n="cohort">코호트</label>
              <select id="contribCohort"></select>
            </div>
            <div class="table-scroll">
              <table id="contribTable"></table>
            </div>
          </div>
        </div>
      </section>
    </div>

    <div class="tab-panel" id="tab-wsi">
      <section class="grid two section">
        <div class="panel">
          <div class="controls">
            <h2 style="margin-right:auto" data-i18n="wsiSmoke">WSI 스모크 테스트</h2>
            <label for="wsiSlide" data-i18n="slide">슬라이드</label>
            <select id="wsiSlide"></select>
          </div>
          <div class="image-frame" id="wsiOverlayFrame"></div>
          <div class="wsi-selected-card" id="wsiSelectedDetails"></div>
        </div>
        <div class="panel">
          <div class="panel-title-row">
            <h2 data-i18n="selectedPatch">선택 패치</h2>
            <button class="small-button" id="enlargePatchButton" type="button" data-i18n="enlargeImage">확대 보기</button>
          </div>
          <div class="image-frame patch-preview-frame" id="wsiPatchPreviewFrame"></div>
          <div class="patch-meta" id="wsiPatchPreviewMeta"></div>
          <h2 style="margin-top:16px" data-i18n="wsiPatchSamples">WSI 패치 샘플</h2>
          <div id="wsiPatchGrid" class="patch-grid"></div>
        </div>
      </section>

      <section class="panel section">
        <h2 data-i18n="wsiFeatures">WSI feature 요약</h2>
        <div class="table-scroll" style="max-height:260px">
          <table id="wsiFeatureTable"></table>
        </div>
      </section>
    </div>

    <p class="footer-note" data-i18n="footerNote">C-index는 교차검증 결과입니다. 유전자 계수는 선택된 feature로 학습한 해석용 모델의 값이므로 1차 생물학적 후보로 해석해야 합니다.</p>
  </main>

  <div class="image-modal" id="imageModal" aria-hidden="true">
    <div class="image-modal-card" role="dialog" aria-modal="true" aria-labelledby="imageModalTitle">
      <div class="image-modal-head">
        <div class="image-modal-title" id="imageModalTitle"></div>
        <button class="small-button" id="imageModalClose" type="button" data-i18n="close">닫기</button>
      </div>
      <div class="image-modal-body" id="imageModalBody"></div>
      <div class="image-modal-foot" id="imageModalMeta"></div>
    </div>
  </div>

  <script>
    const I18N = {
      ko: {
        documentTitle: "TCGA 교종 생존 예측 대시보드",
        appTitle: "TCGA 교종 멀티모달 생존 예측 대시보드",
        appSubtitle: "위험 예측과 약물 재창출 후보 해석을 함께 확인하는 연구용 대시보드",
        language: "언어",
        overviewTab: "개요 & 예측 실행",
        riskPredictionTab: "위험 예측",
        riskInputView: "위험 예측 실행",
        patientDrugView: "환자별 약물 후보",
        performanceTab: "모델 성능 & 생존분석",
        wsiTab: "WSI 병리",
        drugCandidateTab: "약물 후보",
        drugTab: "신약 재창출",
        drugEffectTab: "약물 효과",
        modelGraphTab: "모델 성능 그래프",
        kmGraphTab: "Kaplan-Meier 생존곡선",
        geneGraphTab: "주요 RNA 위험 유전자",
        modelPerformance: "모델 성능 비교",
        viewMode: "보기",
        cohortPerformance: "코호트별 모델 성능",
        wsiEmbeddingComparison: "WSI 임베딩 비교",
        highlightModel: "강조할 모델",
        model: "모델",
        clinical: "임상정보",
        rnaClinical: "RNA + 임상정보",
        wsi: "WSI",
        wsiClinical: "WSI + 임상정보",
        wsiRnaClinical: "WSI + RNA + 임상정보",
        wsiResnetEmbedding: "WSI ResNet18 Embedding",
        wsiResnetEmbeddingRnaClinical: "WSI ResNet18 Embedding + RNA + 임상정보",
        wsiUniEmbedding: "WSI UNI Embedding",
        wsiUniEmbeddingClinical: "WSI UNI Embedding + 임상정보",
        wsiUniEmbeddingRna: "WSI UNI Embedding + RNA",
        wsiUniEmbeddingRnaClinical: "WSI UNI Embedding + RNA + 임상정보",
        wsiConchEmbedding: "WSI CONCH Embedding",
        wsiConchEmbeddingRnaClinical: "WSI CONCH Embedding + RNA + 임상정보",
        coreTakeaway: "핵심 해석",
        inferencePrototype: "RNA + 임상정보 예측 엔진",
        rnaTsv: "RNA TSV",
        clinicalJson: "임상정보 JSON",
        rnaTableUpload: "RNA 표 파일",
        rnaUploadHint: "gene/tpm 컬럼이 있는 TSV 또는 CSV를 드래그하면 아래 셀에 미리 표시됩니다.",
        rnaCellPreview: "RNA 셀 미리보기",
        previewEmpty: "파일을 선택하면 미리보기가 표시됩니다.",
        clinicalCells: "임상정보 입력 셀",
        caseId: "환자 ID",
        sex: "성별",
        age: "진단 시 나이",
        tumorGrade: "종양 등급",
        primaryDiagnosis: "진단명",
        subtype: "분자아형",
        cancerTypeDetailed: "상세 암종",
        tumorType: "종양 유형",
        previewRows: (rows, cols) => `${rows}행 · ${cols}열 미리보기`,
        wsiImageUpload: "WSI 병리 이미지",
        wsiUploadHint: "이번 단계에서는 환자 케이스에 이미지를 함께 등록하고, 실제 WSI 기반 예측은 WSI embedding 모델 학습 후 반영합니다.",
        wsiUploadEmpty: "선택된 WSI 이미지가 없습니다.",
        selectedPatient: "선택 환자",
        selectedPatientEmpty: "예측을 실행하거나 등록 환자를 선택하면 이 영역에 환자 정보가 표시됩니다.",
        linkedWsi: "연결 WSI",
        registeredPatientSearch: "등록 환자 검색",
        drugContextTitle: "선택 환자 기반 약물 후보 해석",
        patientDrugNote: "환자별 RNA 위험 기여 유전자와 약물 타깃 overlap을 이용해 전체 GBM 후보 순위를 재조정했습니다.",
        patientPriorityScore: "개인화 점수",
        patientMatch: "타깃 매칭",
        baseDrugScore: "기본 검증",
        matchedRiskGenes: "매칭 유전자",
        noPatientGeneMatch: "직접 매칭된 위험 유전자는 없어 전체 GBM 후보 근거를 우선 표시합니다.",
        whySelected: "선택 이유",
        targetReason: "타깃 근거",
        clueReason: "발현 반전 근거",
        clinicalReason: "검증 근거",
        predict: "예측 실행",
        useExample: "예시 실행",
        predictionResult: "예측 결과",
        predictionEmpty: "아직 실행된 예측이 없습니다.",
        predictionRunning: "예측 중...",
        predictionFailed: "예측 실패",
        survivalPrediction: "생존 예측 정보",
        noPrediction: "예측 정보 없음",
        selectedPatch: "선택 패치",
        originalPatch: "원본",
        attentionHeatmap: "Attention heatmap",
        enlargeImage: "확대 보기",
        close: "닫기",
        predictedRisk: "예측 위험군",
        riskScore: "위험 점수",
        trainingMedian: "학습 중앙값",
        geneContributions: "유전자 기여도",
        clinicalContributions: "임상정보 기여도",
        missingUpload: "RNA 표 파일을 선택하고 환자 ID를 입력하세요.",
        kmCurve: "Kaplan-Meier 생존곡선",
        run: "실험",
        highestRiskPatients: "고위험 예측 환자",
        topGenes: "주요 RNA 위험 유전자",
        cohort: "코호트",
        analysis: "분석",
        patientContrib: "환자별 유전자 기여도",
        wsiSmoke: "WSI 스모크 테스트",
        slide: "슬라이드",
        wsiPatchSamples: "WSI 패치 샘플",
        wsiFeatures: "WSI feature 요약",
        patchesUsed: "사용 패치",
        selectedWsiPatient: "선택 환자 WSI 수치",
        tissueArea: "조직 면적",
        slidePixels: "슬라이드 크기",
        brightness: "밝기",
        saturation: "염색 강도",
        darkRatio: "어두운 픽셀",
        purpleRatio: "보라색 픽셀",
        pinkRatio: "분홍색 픽셀",
        footerNote: "C-index는 교차검증 결과입니다. 유전자 계수는 선택된 feature로 학습한 해석용 모델의 값이므로 1차 생물학적 후보로 해석해야 합니다.",
        rnaMatrixReady: "RNA 행렬 준비됨",
        rnaMatrixMissing: "RNA 행렬 없음",
        rnaFiles: "RNA 파일",
        wsiFilesIndexed: "WSI 파일 인덱싱됨",
        patients: "환자 수",
        fullCohort: "생존정보가 있는 전체 코호트",
        lowerGradeGlioma: "저등급 교종",
        glioblastoma: "교모세포종",
        rnaMatrix: "RNA 행렬",
        rnaMatrixNote: "681 x 19,938 TPM 표",
        nextModality: "다음 모달리티",
        patchPending: "패치 feature 생성 예정",
        cases: "케이스 수",
        cindex: "C-index",
        logrank: "Log-rank p",
        patient: "환자",
        project: "프로젝트",
        risk: "위험군",
        score: "점수",
        osDays: "전체 생존일",
        status: "상태",
        rank: "순위",
        gene: "유전자",
        feature: "항목",
        direction: "방향",
        contribution: "기여도",
        file: "파일",
        tissueRatio: "조직 비율",
        patches: "패치 수",
        processed: "처리됨",
        noPatches: "패치 없음",
        patchRiskRank: "위험 순위",
        patchRiskScore: "패치 위험점수",
        patchImportance: "중요도",
        timeDays: "시간, 일",
        high: "고위험",
        low: "저위험",
        event: "사망 이벤트",
        censored: "검열",
        raises: "위험 증가",
        lowers: "위험 감소",
        raisesRisk: "위험을 높임",
        lowersRisk: "위험을 낮춤",
        loadError: "대시보드를 불러오지 못했습니다",
        drugRegisterView: "후보 약물 등록",
        drugOverviewView: "후보 순위 개요",
        drugTableView: "통합 후보 테이블",
        drugRegisterTitle: "후보 약물 직접 등록",
        drugRegisterHint: "약물명과 타깃 유전자를 입력하면 통합 후보 테이블과 환자별 약물 후보 계산에 바로 반영됩니다.",
        drugAction: "작용 방향",
        addDrugCandidate: "후보 추가",
        fillDrugExample: "예시 채우기",
        drugExcelUpload: "엑셀/CSV 후보 파일",
        drugUploadHint: "drug_name, primary_target, targets, validation_score 같은 컬럼이 있으면 등록 후보로 바로 반영할 수 있습니다.",
        drugPreviewTitle: "후보 파일 미리보기",
        applyDrugPreview: "미리보기 후보 반영",
        registeredDrugCandidates: "등록된 후보 약물",
        registeredDrugHint: "사용자 등록 후보는 브라우저에 저장되고, 기존 후보와 함께 개인화 약물 후보 계산에 사용됩니다.",
        noUserDrugs: "아직 등록된 사용자 후보가 없습니다.",
        remove: "삭제",
        drugAdded: "후보 약물이 등록되었습니다.",
        drugPreviewApplied: "미리보기 후보를 등록했습니다.",
        drugMissingNameTarget: "약물명과 타깃 유전자는 반드시 입력하세요.",
        drugPreviewMissing: "반영할 후보 파일 미리보기가 없습니다.",
        userRegistered: "사용자 등록",
        drugRankOverview: "재창출 후보 순위",
        coreTargetGenes: "핵심 타깃군 - GBM 차등발현 유전자",
        integratedCandidateTable: "통합 후보 테이블",
        finalDrugShortlist: "최종 재창출 후보 요약",
        candidateValidationMatrix: "후보 검증 매트릭스",
        drugCandidates: "신약 재창출 후보 순위",
        degGenes: "GBM 차등발현 유전자",
        drugCandidateTable: "후보 약물 상세표",
        drugCaution: "정상 뇌 결과가 있으면 GTEx normal brain 비교를 우선 표시하고, 없으면 TCGA-LGG proxy 결과를 표시합니다.",
        gbmExpression: "GBM 발현",
        controlExpression: "비교군 발현",
        drugName: "약물",
        target: "타깃",
        targets: "타깃 목록",
        repurposingScore: "재창출 점수",
        finalRepurposingScore: "최종 점수",
        clueStatus: "CLUE 상태",
        reversalPrior: "반전 근거",
        validationRank: "검증 순위",
        validationTier: "검증 단계",
        validationScore: "검증 점수",
        clueTau: "CLUE tau",
        bbbEstimate: "BBB 추정",
        gbmTrials: "GBM trial",
        evidenceComment: "근거 코멘트",
        log2fc: "log2FC",
        fdr: "FDR",
        approved: "승인",
        antineoplastic: "항암제",
        yes: "예",
        no: "아니오",
        upInGbm: "GBM에서 증가",
        downInGbm: "GBM에서 감소",
        shortlistSummaryText: (summary) => `${summary.comparison || "GBM comparison"} · 후보 ${summary.shortlist_candidates || 0}개 · DEG ${summary.deg_genes_tested || 0}개 · 최상위 후보 ${summary.top_candidate || "없음"} / ${summary.top_candidate_target || ""}`,
        validationSummaryText: (summary) => `PubChem 물성 조회 ${summary.pubchem_fetched || 0}개 · GBM trial flag 후보 ${summary.candidates_with_gbm_trials || 0}개 · 최상위 검증 후보 ${summary.top_validation_candidate || "없음"}`,
        drugSummaryText: (summary) => `${summary.comparison || "GBM comparison"} · GBM ${summary.gbm_samples || 0}명 / 비교군 ${summary.control_samples || 0}명 · DEG ${summary.genes_tested || 0}개 검사 · DGIdb interaction ${summary.dgidb_interactions || 0}개`,
        clueSummaryText: (summary) => `CLUE/LINCS 준비: up ${summary.up_genes || 0}개 / down ${summary.down_genes || 0}개 · 상태 ${summary.clue_status || "not available"}`,
        takeaway: (all, lgg, gbm) => `RNA+임상정보 결합 모델이 전체에서 가장 높고(${all}), LGG 내부에서도 강하게 유지됩니다(${lgg}). GBM-only는 더 어려운 문제(${gbm})라서, WSI를 붙이기 전 중요한 한계점으로 설명할 수 있습니다.`
      },
      en: {
        documentTitle: "TCGA Glioma Survival Dashboard",
        appTitle: "TCGA Glioma Multimodal Survival Dashboard",
        appSubtitle: "Research dashboard for risk prediction and drug repurposing candidate interpretation",
        language: "Language",
        overviewTab: "Overview & Prediction",
        riskPredictionTab: "Risk Prediction",
        riskInputView: "Run Risk Prediction",
        patientDrugView: "Patient Drug Candidates",
        performanceTab: "Model Performance & Survival",
        wsiTab: "WSI Pathology",
        drugCandidateTab: "Drug Candidates",
        drugTab: "Drug Repurposing",
        drugEffectTab: "Drug Effect",
        modelGraphTab: "Model Performance",
        kmGraphTab: "Kaplan-Meier Curve",
        geneGraphTab: "Top RNA Risk Genes",
        modelPerformance: "Model Performance",
        viewMode: "View",
        cohortPerformance: "Cohort Model Performance",
        wsiEmbeddingComparison: "WSI Embedding Comparison",
        highlightModel: "Highlight Model",
        model: "Model",
        clinical: "Clinical",
        rnaClinical: "RNA + Clinical",
        wsi: "WSI",
        wsiClinical: "WSI + Clinical",
        wsiRnaClinical: "WSI + RNA + Clinical",
        wsiResnetEmbedding: "WSI ResNet18 Embedding",
        wsiResnetEmbeddingRnaClinical: "WSI ResNet18 Embedding + RNA + Clinical",
        wsiUniEmbedding: "WSI UNI Embedding",
        wsiUniEmbeddingClinical: "WSI UNI Embedding + Clinical",
        wsiUniEmbeddingRna: "WSI UNI Embedding + RNA",
        wsiUniEmbeddingRnaClinical: "WSI UNI Embedding + RNA + Clinical",
        wsiConchEmbedding: "WSI CONCH Embedding",
        wsiConchEmbeddingRnaClinical: "WSI CONCH Embedding + RNA + Clinical",
        coreTakeaway: "Core Takeaway",
        inferencePrototype: "RNA + Clinical Inference Engine",
        rnaTsv: "RNA TSV",
        clinicalJson: "Clinical JSON",
        rnaTableUpload: "RNA Table File",
        rnaUploadHint: "Drop a TSV or CSV with gene/tpm columns to preview the cells below.",
        rnaCellPreview: "RNA Cell Preview",
        previewEmpty: "A preview will appear after selecting a file.",
        clinicalCells: "Clinical Input Cells",
        caseId: "Patient ID",
        sex: "Sex",
        age: "Age at diagnosis",
        tumorGrade: "Tumor grade",
        primaryDiagnosis: "Primary diagnosis",
        subtype: "Molecular subtype",
        cancerTypeDetailed: "Detailed cancer type",
        tumorType: "Tumor type",
        previewRows: (rows, cols) => `${rows} rows · ${cols} columns previewed`,
        wsiImageUpload: "WSI Pathology Image",
        wsiUploadHint: "At this stage the image is linked to the patient case. WSI-based prediction will be enabled after WSI embedding model training.",
        wsiUploadEmpty: "No WSI image selected.",
        selectedPatient: "Selected Patient",
        selectedPatientEmpty: "Run a prediction or select a registered patient to show patient context here.",
        linkedWsi: "Linked WSI",
        registeredPatientSearch: "Registered Patient Search",
        drugContextTitle: "Patient-linked Drug Candidate Interpretation",
        patientDrugNote: "Whole-cohort GBM drug candidates are re-ranked with patient-level RNA risk-gene and drug-target overlap.",
        patientPriorityScore: "Patient score",
        patientMatch: "Target match",
        baseDrugScore: "Base evidence",
        matchedRiskGenes: "Matched genes",
        noPatientGeneMatch: "No directly matched risk gene was found, so the global GBM candidate evidence is shown first.",
        whySelected: "Why Selected",
        targetReason: "Target Rationale",
        clueReason: "Expression Reversal Evidence",
        clinicalReason: "Validation Evidence",
        predict: "Predict",
        useExample: "Run Example",
        predictionResult: "Prediction Result",
        predictionEmpty: "No prediction has been run yet.",
        predictionRunning: "Running prediction...",
        predictionFailed: "Prediction failed",
        survivalPrediction: "Survival Prediction",
        noPrediction: "No prediction",
        selectedPatch: "Selected Patch",
        originalPatch: "Original",
        attentionHeatmap: "Attention heatmap",
        enlargeImage: "Enlarge",
        close: "Close",
        predictedRisk: "Predicted risk",
        riskScore: "Risk score",
        trainingMedian: "Training median",
        geneContributions: "Gene Contributions",
        clinicalContributions: "Clinical Contributions",
        missingUpload: "Select an RNA table file and enter a patient ID.",
        kmCurve: "Kaplan-Meier Curve",
        run: "Run",
        highestRiskPatients: "Highest Risk Patients",
        topGenes: "Top RNA Risk Genes",
        cohort: "Cohort",
        analysis: "Analysis",
        patientContrib: "Patient Gene Contributions",
        wsiSmoke: "WSI Smoke Test",
        slide: "Slide",
        wsiPatchSamples: "WSI Patch Samples",
        wsiFeatures: "WSI Feature Summary",
        patchesUsed: "Patches used",
        selectedWsiPatient: "Selected WSI Patient Metrics",
        tissueArea: "Tissue area",
        slidePixels: "Slide size",
        brightness: "Brightness",
        saturation: "Stain intensity",
        darkRatio: "Dark pixels",
        purpleRatio: "Purple pixels",
        pinkRatio: "Pink pixels",
        footerNote: "C-index values are cross-validated. Gene coefficients are interpretation models fitted on selected features and should be treated as first-pass biological candidates.",
        rnaMatrixReady: "RNA matrix ready",
        rnaMatrixMissing: "RNA matrix missing",
        rnaFiles: "RNA files",
        wsiFilesIndexed: "WSI files indexed",
        patients: "Patients",
        fullCohort: "Full survival-eligible cohort",
        lowerGradeGlioma: "Lower-grade glioma",
        glioblastoma: "Glioblastoma",
        rnaMatrix: "RNA Matrix",
        rnaMatrixNote: "681 x 19,938 TPM table",
        nextModality: "Next Modality",
        patchPending: "Patch features pending",
        cases: "Cases",
        cindex: "C-index",
        logrank: "Log-rank p",
        patient: "Patient",
        project: "Project",
        risk: "Risk",
        score: "Score",
        osDays: "OS days",
        status: "Status",
        rank: "Rank",
        gene: "Gene",
        feature: "Feature",
        direction: "Direction",
        contribution: "Contribution",
        file: "File",
        tissueRatio: "Tissue ratio",
        patches: "Patches",
        processed: "Processed",
        noPatches: "No patches",
        patchRiskRank: "Risk rank",
        patchRiskScore: "Patch risk score",
        patchImportance: "Importance",
        timeDays: "Time, days",
        high: "High",
        low: "Low",
        event: "Event",
        censored: "Censored",
        raises: "Raises",
        lowers: "Lowers",
        raisesRisk: "Raises risk",
        lowersRisk: "Lowers risk",
        loadError: "Dashboard failed to load",
        drugRegisterView: "Register Candidate",
        drugOverviewView: "Candidate Ranking Overview",
        drugTableView: "Integrated Candidate Table",
        drugRegisterTitle: "Register Candidate Drug",
        drugRegisterHint: "Enter a drug name and target genes to immediately include it in the integrated table and patient-level candidate scoring.",
        drugAction: "Action",
        addDrugCandidate: "Add Candidate",
        fillDrugExample: "Fill Example",
        drugExcelUpload: "Excel/CSV Candidate File",
        drugUploadHint: "Files with columns such as drug_name, primary_target, targets, and validation_score can be previewed and added.",
        drugPreviewTitle: "Candidate File Preview",
        applyDrugPreview: "Apply Preview Candidates",
        registeredDrugCandidates: "Registered Candidate Drugs",
        registeredDrugHint: "User candidates are stored in the browser and used together with existing candidates for patient-level scoring.",
        noUserDrugs: "No user candidates have been registered yet.",
        remove: "Remove",
        drugAdded: "Candidate drug registered.",
        drugPreviewApplied: "Preview candidates registered.",
        drugMissingNameTarget: "Drug name and target gene are required.",
        drugPreviewMissing: "There are no preview candidates to apply.",
        userRegistered: "User registered",
        drugRankOverview: "Repurposing Candidate Ranking",
        coreTargetGenes: "Core Targets - GBM Differential Genes",
        integratedCandidateTable: "Integrated Candidate Table",
        finalDrugShortlist: "Final Repurposing Shortlist",
        candidateValidationMatrix: "Candidate Validation Matrix",
        drugCandidates: "Drug Repurposing Candidates",
        degGenes: "GBM Differential Expression Genes",
        drugCandidateTable: "Candidate Drug Detail",
        drugCaution: "GTEx normal-brain results are shown first when available; otherwise the TCGA-LGG proxy result is shown.",
        gbmExpression: "GBM expression",
        controlExpression: "Control expression",
        drugName: "Drug",
        target: "Target",
        targets: "Targets",
        repurposingScore: "Repurposing score",
        finalRepurposingScore: "Final score",
        clueStatus: "CLUE status",
        reversalPrior: "Reversal prior",
        validationRank: "Validation rank",
        validationTier: "Validation tier",
        validationScore: "Validation score",
        clueTau: "CLUE tau",
        bbbEstimate: "BBB estimate",
        gbmTrials: "GBM trial",
        evidenceComment: "Evidence comment",
        log2fc: "log2FC",
        fdr: "FDR",
        approved: "Approved",
        antineoplastic: "Antineoplastic",
        yes: "Yes",
        no: "No",
        upInGbm: "Up in GBM",
        downInGbm: "Down in GBM",
        shortlistSummaryText: (summary) => `${summary.comparison || "GBM comparison"} · ${summary.shortlist_candidates || 0} candidates · ${summary.deg_genes_tested || 0} DE genes · top ${summary.top_candidate || "none"} / ${summary.top_candidate_target || ""}`,
        validationSummaryText: (summary) => `${summary.pubchem_fetched || 0} PubChem lookups · ${summary.candidates_with_gbm_trials || 0} candidates with GBM trial flags · top validation candidate ${summary.top_validation_candidate || "none"}`,
        drugSummaryText: (summary) => `${summary.comparison || "GBM comparison"} · GBM ${summary.gbm_samples || 0} / control ${summary.control_samples || 0} · ${summary.genes_tested || 0} genes tested · ${summary.dgidb_interactions || 0} DGIdb interactions`,
        clueSummaryText: (summary) => `CLUE/LINCS ready: ${summary.up_genes || 0} up / ${summary.down_genes || 0} down genes · status ${summary.clue_status || "not available"}`,
        takeaway: (all, lgg, gbm) => `The combined RNA + clinical model is strongest overall (${all}) and remains useful within LGG (${lgg}). GBM-only is harder (${gbm}), which is a key limitation to discuss before WSI is added.`
      }
    };

    const COHORT_LABELS = {
      ko: { "All LGG+GBM": "전체 LGG+GBM", "LGG-only": "LGG만", "GBM-only": "GBM만", "WSI dev 50": "WSI dev 50" },
      en: { "All LGG+GBM": "All LGG+GBM", "LGG-only": "LGG-only", "GBM-only": "GBM-only", "WSI dev 50": "WSI dev 50" }
    };
    const MODEL_LABELS = {
      ko: {
        Clinical: "임상정보",
        RNA: "RNA",
        "RNA + Clinical": "RNA + 임상정보",
        WSI: "WSI",
        "WSI + Clinical": "WSI + 임상정보",
        "WSI + RNA + Clinical": "WSI + RNA + 임상정보",
        "WSI ResNet18 Embedding": "WSI ResNet18 Embedding",
        "WSI ResNet18 Embedding + RNA + Clinical": "WSI ResNet18 Embedding + RNA + 임상정보",
        "WSI UNI Embedding": "WSI UNI Embedding",
        "WSI UNI Embedding + Clinical": "WSI UNI Embedding + 임상정보",
        "WSI UNI Embedding + RNA": "WSI UNI Embedding + RNA",
        "WSI UNI Embedding + RNA + Clinical": "WSI UNI Embedding + RNA + 임상정보",
        "WSI CONCH Embedding": "WSI CONCH Embedding",
        "WSI CONCH Embedding + RNA + Clinical": "WSI CONCH Embedding + RNA + 임상정보"
      },
      en: {
        Clinical: "Clinical",
        RNA: "RNA",
        "RNA + Clinical": "RNA + Clinical",
        WSI: "WSI",
        "WSI + Clinical": "WSI + Clinical",
        "WSI + RNA + Clinical": "WSI + RNA + Clinical",
        "WSI ResNet18 Embedding": "WSI ResNet18 Embedding",
        "WSI ResNet18 Embedding + RNA + Clinical": "WSI ResNet18 Embedding + RNA + Clinical",
        "WSI UNI Embedding": "WSI UNI Embedding",
        "WSI UNI Embedding + Clinical": "WSI UNI Embedding + Clinical",
        "WSI UNI Embedding + RNA": "WSI UNI Embedding + RNA",
        "WSI UNI Embedding + RNA + Clinical": "WSI UNI Embedding + RNA + Clinical",
        "WSI CONCH Embedding": "WSI CONCH Embedding",
        "WSI CONCH Embedding + RNA + Clinical": "WSI CONCH Embedding + RNA + Clinical"
      }
    };
    const ANALYSIS_LABELS = {
      ko: { "RNA-only interpretation": "RNA 단독 해석", "RNA adjusted for clinical": "임상정보 보정 RNA 해석" },
      en: { "RNA-only interpretation": "RNA-only interpretation", "RNA adjusted for clinical": "RNA adjusted for clinical" }
    };
    const RUN_META = {
      all_rna_clinical: ["All LGG+GBM", "RNA + Clinical"],
      all_rna: ["All LGG+GBM", "RNA"],
      all_clinical: ["All LGG+GBM", "Clinical"],
      lgg_rna_clinical: ["LGG-only", "RNA + Clinical"],
      lgg_rna: ["LGG-only", "RNA"],
      lgg_clinical: ["LGG-only", "Clinical"],
      gbm_rna_clinical: ["GBM-only", "RNA + Clinical"],
      gbm_rna: ["GBM-only", "RNA"],
      gbm_clinical: ["GBM-only", "Clinical"]
    };
    const runOptions = Object.keys(RUN_META);
    const contribCohorts = [["all", "All LGG+GBM"], ["lgg", "LGG-only"], ["gbm", "GBM-only"]];
    const state = {
      summary: null,
      comparison: [],
      genes: [],
      km: [],
      predictions: [],
      wsiRiskPredictions: [],
      contrib: [],
      contribAll: [],
      contribGbm: [],
      drugs: { summary: {}, candidates: [], deg: [] },
      userDrugCandidates: [],
      drugPreviewRows: [],
      wsi: { slides: [], patches: [] },
      wsiFeatures: [],
      selectedPerformanceKey: null,
      selectedDrugView: "register",
      selectedDrugName: null,
      selectedWsiFileId: null,
      selectedWsiPatchKey: null,
      selectedWsiPatchUrl: null,
      selectedWsiPatch: null,
      activePatient: null,
      pendingWsi: null,
      drugPatientSearch: ""
    };

    function lang() {
      const selected = document.getElementById("languageSelect")?.value || "ko";
      return I18N[selected] ? selected : "ko";
    }

    function t(key) {
      return I18N[lang()][key] || I18N.en[key] || key;
    }

    function cohortLabel(value) {
      return COHORT_LABELS[lang()][value] || value;
    }

    function modelLabel(value) {
      return MODEL_LABELS[lang()][value] || value;
    }

    function escapeSvgText(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function chartModelLabel(value) {
      const labels = {
        ko: {
          "WSI ResNet18 Embedding": "ResNet18 임베딩",
          "WSI ResNet18 Embedding + RNA + Clinical": "ResNet18 + RNA + 임상",
          "WSI UNI Embedding": "UNI 임베딩",
          "WSI UNI Embedding + Clinical": "UNI + 임상",
          "WSI UNI Embedding + RNA": "UNI + RNA",
          "WSI UNI Embedding + RNA + Clinical": "UNI + RNA + 임상",
          "WSI CONCH Embedding": "CONCH 임베딩",
          "WSI CONCH Embedding + RNA + Clinical": "CONCH + RNA + 임상"
        },
        en: {
          "WSI ResNet18 Embedding": "ResNet18 Embedding",
          "WSI ResNet18 Embedding + RNA + Clinical": "ResNet18 + RNA + Clinical",
          "WSI UNI Embedding": "UNI Embedding",
          "WSI UNI Embedding + Clinical": "UNI + Clinical",
          "WSI UNI Embedding + RNA": "UNI + RNA",
          "WSI UNI Embedding + RNA + Clinical": "UNI + RNA + Clinical",
          "WSI CONCH Embedding": "CONCH Embedding",
          "WSI CONCH Embedding + RNA + Clinical": "CONCH + RNA + Clinical"
        }
      };
      return labels[lang()][value] || modelLabel(value);
    }

    function chartLabelLines(row, mode) {
      if (mode === "wsi") {
        const label = chartModelLabel(row.model);
        const parts = label.split(" + ");
        if (parts.length > 1) return [parts[0], `+ ${parts.slice(1).join(" + ")}`];
        return [label];
      }
      return [`${cohortLabel(row.cohort)} · ${modelLabel(row.model)}`];
    }

    function analysisLabel(value) {
      return ANALYSIS_LABELS[lang()][value] || value;
    }

    function riskLabel(value) {
      if (value === "High") return t("high");
      if (value === "Low") return t("low");
      return value;
    }

    function directionLabel(value) {
      return value === "raises_risk" ? t("raises") : t("lowers");
    }

    function patchDirectionLabel(value) {
      if (value === "raises_risk") return t("raisesRisk");
      if (value === "lowers_risk") return t("lowersRisk");
      return "";
    }

    function patchRiskClass(value) {
      if (value === "raises_risk") return "raises-risk";
      if (value === "lowers_risk") return "lowers-risk";
      return "";
    }

    function runLabel(key) {
      const [cohort, model] = RUN_META[key];
      return `${cohortLabel(cohort)} - ${modelLabel(model)}`;
    }

    function activePatientPrediction() {
      return state.activePatient?.prediction || null;
    }

    function activePatientCaseId() {
      return activePatientPrediction()?.case_submitter_id || state.activePatient?.clinical?.case_submitter_id || "";
    }

    function contributionGene(row) {
      return String(row?.gene || row?.feature || "").trim().toUpperCase();
    }

    function contributionValue(row) {
      return Number(row?.risk_contribution ?? row?.contribution ?? 0);
    }

    function patientContribForCase(caseId, projectId = "") {
      if (!caseId) return [];
      const sources = projectId === "TCGA-GBM"
        ? [["gbm", state.contribGbm], ["all", state.contribAll]]
        : [["all", state.contribAll], ["gbm", state.contribGbm]];
      const seen = new Set();
      const rows = [];
      sources.forEach(([source, data]) => {
        (data || [])
          .filter(row => row.case_submitter_id === caseId)
          .forEach(row => {
            const gene = contributionGene(row);
            if (!gene || seen.has(gene)) return;
            seen.add(gene);
            rows.push({ ...row, feature: row.feature || row.gene, gene: row.gene || row.feature, source });
          });
      });
      return rows.sort((a, b) => Math.abs(contributionValue(b)) - Math.abs(contributionValue(a)));
    }

    const fmt = (value, digits = 3) => {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : "";
    };

    const fmtSmall = (value, digits = 3) => {
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      if (Math.abs(n) > 0 && Math.abs(n) < 0.001) return n.toExponential(2);
      return n.toFixed(digits);
    };

    async function getJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(url);
      return response.json();
    }

    function fillSelect(id, options) {
      const select = document.getElementById(id);
      const previous = select.value;
      select.innerHTML = options.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
      if (options.some(([value]) => value === previous)) select.value = previous;
    }

    function applyStaticText() {
      document.documentElement.lang = lang();
      document.title = t("documentTitle");
      document.querySelectorAll("[data-i18n]").forEach(node => {
        node.textContent = t(node.dataset.i18n);
      });
    }

    function refreshSelectLabels() {
      fillSelect("kmRun", runOptions.map(key => [key, runLabel(key)]));
      fillSelect("predRun", runOptions.map(key => [key, runLabel(key)]));
      fillSelect("contribCohort", contribCohorts.map(([key, cohort]) => [key, cohortLabel(cohort)]));
      const cohorts = [...new Set(state.genes.map(r => r.cohort))];
      const analyses = [...new Set(state.genes.map(r => r.analysis))];
      fillSelect("geneCohort", cohorts.map(v => [v, cohortLabel(v)]));
      fillSelect("geneAnalysis", analyses.map(v => [v, analysisLabel(v)]));
    }

    function activateTab(tabKey) {
      document.querySelectorAll(".tab-button").forEach(button => {
        const active = button.dataset.tabTarget === tabKey;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `tab-${tabKey}`);
      });
    }

    function activateAnalysisTab(tabKey) {
      document.querySelectorAll("[data-analysis-target]").forEach(button => {
        const active = button.dataset.analysisTarget === tabKey;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      ["model", "km", "genes"].forEach(key => {
        document.getElementById(`analysis-${key}`)?.classList.toggle("active", key === tabKey);
        document.getElementById(`analysis-${key}-detail`)?.classList.toggle("active", key === tabKey);
      });
    }

    function activateRiskView(viewKey) {
      document.querySelectorAll("[data-risk-view-target]").forEach(button => {
        const active = button.dataset.riskViewTarget === viewKey;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".risk-view-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `risk-view-${viewKey}`);
      });
    }

    function activateDrugView(viewKey) {
      state.selectedDrugView = viewKey;
      document.querySelectorAll("[data-drug-view-target]").forEach(button => {
        const active = button.dataset.drugViewTarget === viewKey;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".drug-view-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `drug-view-${viewKey}`);
      });
    }

    function activePatientMetricsHtml(mode = "default") {
      const patient = state.activePatient;
      const prediction = patient?.prediction;
      if (!patient || !prediction) {
        return `<div class="hint">${t("selectedPatientEmpty")}</div>`;
      }
      const riskClass = prediction.risk_group === "High" ? "risk-high" : "risk-low";
      const wsiName = patient.wsi?.name || patient.wsi?.file_name || t("noPrediction");
      const metrics = [
        [t("risk"), `<span class="${riskClass}">${riskLabel(prediction.risk_group)}</span>`],
        [t("score"), fmt(prediction.risk_score, 3)],
        [t("project"), prediction.project_id || patient.clinical?.project_id || ""],
        [t("linkedWsi"), wsiName],
      ];
      const metricHtml = metrics.map(([label, value]) => `
        <div class="patient-context-metric">
          <div>${label}</div>
          <div>${value}</div>
        </div>
      `).join("");
      const genes = (patient.gene_contributions || []).slice(0, 5).map(row => row.feature).filter(Boolean).join(", ");
      const note = mode === "drug"
        ? `<p class="hint">${t("patientDrugNote")}${genes ? ` ${t("geneContributions")}: ${genes}` : ""}</p>`
        : "";
      const drugCards = mode === "drug" ? patientDrugCardsHtml() : "";
      const preview = mode === "wsi" && patient.wsi?.url
        ? `<div class="uploaded-wsi-preview"><img src="${patient.wsi.url}" alt="${activePatientCaseId()} uploaded WSI"></div>`
        : "";
      return `
        <div class="patient-context-head">
          <div>
            <div class="patient-context-title">${mode === "drug" ? t("drugContextTitle") : t("selectedPatient")} · ${activePatientCaseId()}</div>
            <div class="patient-context-subtitle">${t("survivalPrediction")} · ${prediction.model_dir || "RNA + Clinical"}</div>
          </div>
          <span class="pill">${riskLabel(prediction.risk_group)}</span>
        </div>
        <div class="patient-context-grid">${metricHtml}</div>
        ${note}
        ${drugCards}
        ${preview}
      `;
    }

    function patientDrugCardsHtml() {
      const rows = personalizedDrugRows().slice(0, 3);
      if (!rows.length) return "";
      return `
        <h3>${t("whySelected")}</h3>
        <div class="patient-drug-cards">
          ${rows.map(row => {
            const tau = Number(row.clue_tau_score);
            const tauText = Number.isFinite(tau) ? `CLUE tau ${fmt(tau, 2)}` : "CLUE 미확인";
            const matched = row.patient_matched_genes || [];
            const matchText = matched.length
              ? `${t("matchedRiskGenes")}: ${matched.map(item => `${item.gene} ${fmt(item.contribution, 3)}`).join(", ")}.`
              : t("noPatientGeneMatch");
            const reason = `${row.primary_target || ""} 타깃 후보. ${matchText} ${row.validation_comment_ko || row.rationale_ko || ""} ${tauText}.`;
            const matchHtml = matched.length
              ? `<div class="patient-drug-match">${matched.slice(0, 4).map(item => `<span>${item.gene} ${fmt(item.contribution, 2)}</span>`).join("")}</div>`
              : "";
            return `
              <div class="patient-drug-card">
                <div class="patient-drug-card-title">
                  <strong>${row.drug_name}</strong>
                  <span>${fmt(row.patient_priority_score, 3)}</span>
                </div>
                ${matchHtml}
                <p>${reason}</p>
                <div class="patient-drug-scoreline">
                  <div><span>${t("patientPriorityScore")}</span><strong>${fmt(row.patient_priority_score, 3)}</strong></div>
                  <div><span>${t("patientMatch")}</span><strong>${fmt(row.patient_match_score, 3)}</strong></div>
                  <div><span>${t("baseDrugScore")}</span><strong>${fmt(row.patient_base_score, 3)}</strong></div>
                </div>
              </div>
            `;
          }).join("")}
        </div>
      `;
    }

    function mergedDrugRows() {
      const validationRows = [...(state.userDrugCandidates || []), ...(state.drugs?.validation_matrix || [])];
      const candidateRowsRaw = [...(state.drugs?.candidates || []), ...(state.userDrugCandidates || [])];
      const validationByName = new Map(validationRows.map(row => [row.drug_name, row]));
      const candidateRows = candidateRowsRaw.map(row => ({ ...row, ...(validationByName.get(row.drug_name) || {}) }));
      const seen = new Set(candidateRows.map(row => row.drug_name));
      const validationOnly = validationRows.filter(row => !seen.has(row.drug_name));
      return [...candidateRows, ...validationOnly];
    }

    function drugTargetSet(row) {
      const values = [row.primary_target, row.target, row.targets]
        .filter(Boolean)
        .flatMap(value => String(value).replace(/[\[\]'"]/g, "").split(/[;,]/))
        .map(value => value.trim().toUpperCase())
        .filter(Boolean);
      return new Set(values);
    }

    function personalizedDrugRows() {
      const patient = state.activePatient;
      const prediction = patient?.prediction || {};
      const patientContrib = (patient?.gene_contributions || []).filter(row => contributionGene(row));
      const positiveSignals = patientContrib
        .map(row => Math.max(0, contributionValue(row)))
        .filter(value => Number.isFinite(value));
      const maxSignal = Math.max(...positiveSignals, 0.001);
      const riskBonus = prediction.risk_group === "High" ? 1 : 0.35;

      return mergedDrugRows().map(row => {
        const targets = drugTargetSet(row);
        const matched = patientContrib
          .filter(geneRow => targets.has(contributionGene(geneRow)))
          .map(geneRow => ({
            gene: contributionGene(geneRow),
            contribution: contributionValue(geneRow),
            direction: geneRow.direction || "",
          }))
          .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
        const matchSignal = matched.reduce((sum, item) => sum + Math.max(0, item.contribution), 0);
        const matchScore = Math.min(1, Math.max(0, matchSignal / maxSignal));
        const baseScore = Number(row.validation_score ?? row.final_repurposing_score ?? row.repurposing_score ?? 0);
        const safeBase = Number.isFinite(baseScore) ? Math.max(0, Math.min(1, baseScore)) : 0;
        const priorityScore = (safeBase * 0.5) + (matchScore * 0.4) + (riskBonus * 0.1);
        return {
          ...row,
          patient_base_score: safeBase,
          patient_match_score: matchScore,
          patient_priority_score: priorityScore,
          patient_matched_genes: matched,
        };
      }).sort((a, b) => {
        const scoreDiff = Number(b.patient_priority_score || 0) - Number(a.patient_priority_score || 0);
        if (Math.abs(scoreDiff) > 0.0001) return scoreDiff;
        return Number(b.patient_match_score || 0) - Number(a.patient_match_score || 0);
      });
    }

    function renderPatientContexts() {
      const targets = [
        ["drugPatientContext", "drug"],
      ];
      targets.forEach(([id, mode]) => {
        const node = document.getElementById(id);
        if (node) node.innerHTML = activePatientMetricsHtml(mode);
      });
    }

    function matchingWsiForCase(caseId) {
      return (state.wsi.slides || []).find(slide => slide.case_submitter_id === caseId) || null;
    }

    function setActivePatient(patient) {
      state.activePatient = patient;
      const caseId = activePatientCaseId();
      const matchedSlide = matchingWsiForCase(caseId);
      if (matchedSlide && !patient.wsi?.url) {
        state.selectedWsiFileId = matchedSlide.file_id;
        state.activePatient.wsi = {
          name: matchedSlide.file_name || matchedSlide.file_id,
          file_name: matchedSlide.file_name,
          file_id: matchedSlide.file_id,
        };
      }
      if (state.activePatient && !(state.activePatient.gene_contributions || []).length) {
        state.activePatient.gene_contributions = patientContribForCase(
          caseId,
          state.activePatient.prediction?.project_id || state.activePatient.clinical?.project_id || ""
        );
      }
      renderPatientContexts();
      renderDrugPatientList();
      if (state.wsi.slides.length) renderWsi();
      if (state.wsiFeatures.length) renderWsiFeatures(state.wsiFeatures);
    }

    function setActivePatientFromPrediction(payload, clinical) {
      const prediction = payload.prediction || {};
      prediction.project_id = prediction.project_id || clinical?.project_id || "";
      setActivePatient({
        prediction,
        clinical,
        wsi: state.pendingWsi,
        gene_contributions: payload.gene_contributions || [],
        clinical_contributions: payload.clinical_contributions || [],
      });
    }

    function renderStats(summary) {
      document.getElementById("statusPills").innerHTML = [
        summary.rna_matrix_exists ? t("rnaMatrixReady") : t("rnaMatrixMissing"),
        `${summary.rna_files} ${t("rnaFiles")}`,
        `${summary.wsi_files} ${t("wsiFilesIndexed")}`
      ].map(text => `<span class="pill">${text}</span>`).join("");

      const stats = [
        [t("patients"), summary.patients, t("fullCohort")],
        ["LGG", summary.lgg, t("lowerGradeGlioma")],
        ["GBM", summary.gbm, t("glioblastoma")],
        [t("rnaMatrix"), `${summary.rna_matrix_size_mb} MB`, t("rnaMatrixNote")],
        [t("nextModality"), "WSI dev 50", `${state.wsi?.slides?.length || 0} ${t("patches")}`]
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value, note]) => `
        <div class="panel metric">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
          <div class="note">${note}</div>
        </div>
      `).join("");
    }

    function colorForModel(model) {
      if (model === "Clinical") return "#276fbf";
      if (model === "RNA") return "#257a52";
      if (model === "RNA + Clinical") return "#c44536";
      if (model === "WSI") return "#ad741f";
      if (model === "WSI + Clinical") return "#6f5aa7";
      if (model === "WSI ResNet18 Embedding") return "#cc6f32";
      if (model === "WSI ResNet18 Embedding + RNA + Clinical") return "#0f766e";
      if (model === "WSI UNI Embedding") return "#9357c8";
      if (model === "WSI UNI Embedding + Clinical") return "#7c3aed";
      if (model === "WSI UNI Embedding + RNA") return "#6d28d9";
      if (model === "WSI UNI Embedding + RNA + Clinical") return "#5b3aa4";
      if (model === "WSI CONCH Embedding") return "#d14d72";
      if (model === "WSI CONCH Embedding + RNA + Clinical") return "#9d174d";
      return "#1f7a7a";
    }

    function modelOrder(model) {
      const order = [
        "Clinical",
        "RNA",
        "RNA + Clinical",
        "WSI",
        "WSI + Clinical",
        "WSI + RNA + Clinical",
        "WSI ResNet18 Embedding",
        "WSI ResNet18 Embedding + RNA + Clinical",
        "WSI UNI Embedding",
        "WSI UNI Embedding + Clinical",
        "WSI UNI Embedding + RNA",
        "WSI UNI Embedding + RNA + Clinical",
        "WSI CONCH Embedding",
        "WSI CONCH Embedding + RNA + Clinical",
      ];
      const index = order.indexOf(model);
      return index >= 0 ? index : 999;
    }

    function performanceKey(row) {
      return `${row.cohort}::${row.model}`;
    }

    function performanceRowsForMode(rows) {
      const mode = document.getElementById("performanceMode")?.value || "cohort";
      const cohortModels = new Set(["Clinical", "RNA", "RNA + Clinical"]);
      const filtered = rows.filter(row => {
        if (mode === "wsi") return row.cohort === "WSI dev 50";
        return row.cohort !== "WSI dev 50" && cohortModels.has(row.model);
      });
      return filtered.sort((a, b) => {
        const cohort = String(a.cohort).localeCompare(String(b.cohort));
        return cohort || modelOrder(a.model) - modelOrder(b.model);
      });
    }

    function selectPerformance(rowKey) {
      state.selectedPerformanceKey = rowKey;
      renderPerformance(state.comparison);
    }

    function renderPerformanceCells(rows) {
      const mode = document.getElementById("performanceMode")?.value || "cohort";
      const cells = rows.map(row => {
        const key = performanceKey(row);
        const selected = key === state.selectedPerformanceKey;
        const displayName = mode === "wsi" ? chartModelLabel(row.model) : modelLabel(row.model);
        return `
          <button class="performance-cell ${selected ? "is-selected" : ""}" type="button" data-performance-key="${key}">
            <span>
              <span class="performance-cell-name" title="${modelLabel(row.model)}">${displayName}</span>
              <span class="performance-cell-meta">${cohortLabel(row.cohort)} · n=${row.n_cases}</span>
            </span>
            <span class="performance-cell-value">${fmt(row.c_index_cv, 3)}</span>
          </button>
        `;
      }).join("");
      document.getElementById("performanceCells").innerHTML = cells;
    }

    function renderPerformance(rows) {
      const displayRows = performanceRowsForMode(rows);
      if (!displayRows.length) {
        document.getElementById("performanceChart").innerHTML = `<span class="hint">${t("noPrediction")}</span>`;
        document.getElementById("performanceCells").innerHTML = "";
        return;
      }
      if (!state.selectedPerformanceKey || !displayRows.some(row => performanceKey(row) === state.selectedPerformanceKey)) {
        state.selectedPerformanceKey = performanceKey(displayRows[0]);
      }
      const mode = document.getElementById("performanceMode")?.value || "cohort";
      const rowGap = mode === "wsi" ? 38 : 30;
      const width = 900, height = Math.max(300, 72 + displayRows.length * rowGap), left = mode === "wsi" ? 260 : 210, right = 58, top = 22, bottom = 38;
      const innerW = width - left - right;
      const barH = 18;
      let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">`;
      for (let i = 0; i <= 5; i++) {
        const x = left + innerW * (i / 5);
        const val = (i / 5).toFixed(1);
        svg += `<line x1="${x}" y1="${top}" x2="${x}" y2="${height-bottom}" stroke="#e8ecf1"/>`;
        svg += `<text x="${x}" y="${height-14}" font-size="11" text-anchor="middle" fill="#617080">${val}</text>`;
      }
      displayRows.forEach((row, index) => {
        const key = performanceKey(row);
        const selected = key === state.selectedPerformanceKey;
        const y = top + 12 + index * rowGap;
        const label = `${cohortLabel(row.cohort)} · ${modelLabel(row.model)}`;
        const labelLines = chartLabelLines(row, mode);
        const w = Math.max(0, Math.min(1, Number(row.c_index_cv))) * innerW;
        const opacity = selected ? "1" : "0.34";
        const stroke = selected ? `stroke="#1a1c21" stroke-width="1.4"` : "";
        svg += `<text x="8" y="${labelLines.length > 1 ? y + 7 : y + 13}" font-size="12" font-weight="${selected ? "800" : "500"}" fill="#1f2933" opacity="${opacity}"><title>${escapeSvgText(label)}</title>${labelLines.map((line, lineIndex) => `<tspan x="8" dy="${lineIndex === 0 ? 0 : 13}">${escapeSvgText(line)}</tspan>`).join("")}</text>`;
        svg += `<rect x="${left}" y="${y}" width="${w}" height="${barH}" fill="${colorForModel(row.model)}" opacity="${opacity}" rx="3" ${stroke}/>`;
        svg += `<text x="${left + w + 6}" y="${y + 13}" font-size="12" font-weight="${selected ? "800" : "500"}" fill="#1f2933" opacity="${opacity}">${fmt(row.c_index_cv)}</text>`;
      });
      svg += `</svg>`;
      document.getElementById("performanceChart").innerHTML = svg;
      renderPerformanceCells(displayRows);
    }

    function renderComparisonTable(rows) {
      const htmlRows = rows.map(r => `
        <tr>
          <td>${cohortLabel(r.cohort)}</td>
          <td>${modelLabel(r.model)}</td>
          <td>${r.n_cases}</td>
          <td>${fmt(r.c_index_cv, 4)}</td>
          <td>${Number(r.logrank_p).toExponential(2)}</td>
        </tr>
      `).join("");
      document.getElementById("comparisonTable").innerHTML = `
        <thead><tr><th>${t("cohort")}</th><th>${t("model")}</th><th>${t("cases")}</th><th>${t("cindex")}</th><th>${t("logrank")}</th></tr></thead>
        <tbody>${htmlRows}</tbody>
      `;
      const allCombo = rows.find(r => r.cohort === "All LGG+GBM" && r.model === "RNA + Clinical");
      const lggCombo = rows.find(r => r.cohort === "LGG-only" && r.model === "RNA + Clinical");
      const gbmCombo = rows.find(r => r.cohort === "GBM-only" && r.model === "RNA + Clinical");
      document.getElementById("takeaway").textContent =
        t("takeaway")(fmt(allCombo.c_index_cv), fmt(lggCombo.c_index_cv), fmt(gbmCombo.c_index_cv));
    }

    function renderKm(rows) {
      const width = 720, height = 310, left = 52, right = 22, top = 18, bottom = 42;
      const innerW = width - left - right;
      const innerH = height - top - bottom;
      const maxT = Math.max(...rows.map(r => Number(r.timeline))) || 1;
      const groups = [...new Set(rows.map(r => r.risk_group))];
      const colors = { High: "#c44536", Low: "#276fbf" };
      const x = t => left + (Number(t) / maxT) * innerW;
      const y = s => top + (1 - Number(s)) * innerH;
      let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" role="img">`;
      svg += `<line x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}" stroke="#303030"/>`;
      svg += `<line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" stroke="#303030"/>`;
      [0, 0.5, 1].forEach(v => {
        svg += `<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="#eef1f5"/>`;
        svg += `<text x="${left-10}" y="${y(v)+4}" font-size="11" text-anchor="end" fill="#617080">${v.toFixed(1)}</text>`;
      });
      groups.forEach(group => {
        const pts = rows.filter(r => r.risk_group === group).sort((a,b) => Number(a.timeline)-Number(b.timeline));
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.timeline).toFixed(1)},${y(p.survival_probability).toFixed(1)}`).join(" ");
        svg += `<path d="${d}" fill="none" stroke="${colors[group] || "#257a52"}" stroke-width="2.5"/>`;
        const last = pts[pts.length - 1];
        if (last) svg += `<text x="${x(last.timeline)-8}" y="${y(last.survival_probability)-6}" font-size="12" text-anchor="end" fill="${colors[group] || "#257a52"}">${riskLabel(group)}</text>`;
      });
      svg += `<text x="${left + innerW / 2}" y="${height-8}" text-anchor="middle" font-size="12" fill="#617080">${t("timeDays")}</text>`;
      svg += `</svg>`;
      document.getElementById("kmChart").innerHTML = svg;
    }

    function renderPredictions(rows) {
      const htmlRows = rows.slice(0, 12).map(r => `
        <tr>
          <td>${r.case_submitter_id}</td>
          <td>${r.project_id}</td>
          <td>${riskLabel(r.risk_group)}</td>
          <td>${fmt(r.risk_score, 3)}</td>
          <td>${Math.round(Number(r.os_days))}</td>
          <td>${Number(r.os_event) ? t("event") : t("censored")}</td>
        </tr>
      `).join("");
      document.getElementById("predictionTable").innerHTML = `
        <thead><tr><th>${t("patient")}</th><th>${t("project")}</th><th>${t("risk")}</th><th>${t("score")}</th><th>${t("osDays")}</th><th>${t("status")}</th></tr></thead>
        <tbody>${htmlRows}</tbody>
      `;
    }

    function renderGenes(rows) {
      const cohort = document.getElementById("geneCohort").value;
      const analysis = document.getElementById("geneAnalysis").value;
      const data = rows
        .filter(r => r.cohort === cohort && r.analysis === analysis)
        .sort((a, b) => Math.abs(Number(b.coef)) - Math.abs(Number(a.coef)))
        .slice(0, 12);
      const maxAbs = Math.max(...data.map(r => Math.abs(Number(r.coef))), 0.001);
      document.getElementById("geneBars").innerHTML = data.map(r => {
        const value = Number(r.coef);
        const w = Math.max(1, Math.abs(value) / maxAbs * 50);
        const cls = value >= 0 ? "pos" : "neg";
        const title = `${value >= 0 ? t("raisesRisk") : t("lowersRisk")}; HR ${fmt(r.hazard_ratio, 3)}`;
        return `
          <div class="gene-row">
            <strong>${r.gene}</strong>
            <div class="bar-track" title="${title}">
              <span class="bar-zero"></span>
              <span class="bar ${cls}" style="width:${w}%"></span>
            </div>
            <span>${fmt(value, 3)}</span>
          </div>
        `;
      }).join("");
    }

    function degDirectionLabel(value) {
      if (value === "up_in_gbm") return t("upInGbm");
      if (value === "down_in_gbm") return t("downInGbm");
      return value || "";
    }

    function isTruthy(value) {
      return value === true || value === "True" || value === "true" || value === 1 || value === "1";
    }

    function yesNo(value) {
      return isTruthy(value) ? t("yes") : t("no");
    }

    function parseMaybeNumber(value, fallback = null) {
      const n = Number(value);
      return Number.isFinite(n) ? n : fallback;
    }

    function canonicalDrugName(value) {
      return String(value || "").trim().toUpperCase();
    }

    function splitTargets(value) {
      return String(value || "")
        .replace(/[\[\]'"]/g, "")
        .split(/[;,]/)
        .map(item => item.trim().toUpperCase())
        .filter(Boolean);
    }

    function normalizeUserDrugCandidate(raw) {
      const drugName = raw.drug_name || raw.drug || raw.Drug || raw["약물명"] || raw["약물"] || "";
      const primaryTarget = raw.primary_target || raw.target || raw.Target || raw["타깃"] || raw["타겟"] || "";
      const targets = raw.targets || raw.target_genes || raw.Targets || raw["타깃 목록"] || raw["타겟 목록"] || primaryTarget;
      const validationScore = parseMaybeNumber(raw.validation_score ?? raw.score ?? raw["검증 점수"], 0.5);
      const clueTau = parseMaybeNumber(raw.clue_tau_score ?? raw.clue_tau ?? raw["CLUE tau"], null);
      const bbbLabel = raw.bbb_rule_label || raw.bbb || raw["BBB 추정"] || "미확인";
      const trialCount = parseMaybeNumber(raw.gbm_trial_count ?? raw.gbm_trials ?? raw["GBM trial"], 0);
      return {
        validation_rank: parseMaybeNumber(raw.validation_rank ?? raw.rank ?? raw["검증 순위"], null),
        validation_tier: t("userRegistered"),
        drug_name: String(drugName).trim(),
        primary_target: String(primaryTarget || targets).split(/[;,]/)[0].trim().toUpperCase(),
        targets: splitTargets(targets || primaryTarget).join(";"),
        interaction_types: raw.interaction_types || raw.action || raw["작용 방향"] || "unknown",
        validation_score: Math.max(0, Math.min(1, validationScore || 0)),
        final_repurposing_score: Math.max(0, Math.min(1, validationScore || 0)),
        repurposing_score: Math.max(0, Math.min(1, validationScore || 0)),
        clue_tau_score: clueTau,
        clue_status: Number.isFinite(clueTau) ? "user_entered" : "not_provided",
        clue_connectivity_direction: Number.isFinite(clueTau) ? (clueTau < 0 ? "reversal" : "mimicry") : "",
        bbb_rule_label: bbbLabel,
        bbb_rule_score: bbbLabel.includes("우호") ? 1 : bbbLabel.includes("경계") ? 0.6 : bbbLabel.includes("불리") ? 0 : null,
        gbm_trial_count: Math.max(0, trialCount || 0),
        approved: isTruthy(raw.approved ?? raw["승인"]),
        antineoplastic: isTruthy(raw.antineoplastic ?? raw["항암제"]),
        validation_comment_ko: raw.validation_comment_ko || raw.evidence_comment || raw.memo || raw["근거 코멘트"] || "사용자가 등록한 후보입니다. 환자별 위험 유전자와 타깃 매칭을 통해 우선순위를 재계산합니다.",
        source: "user",
      };
    }

    function validUserDrugCandidate(row) {
      return row.drug_name && (row.primary_target || row.targets);
    }

    function saveUserDrugCandidates() {
      localStorage.setItem("tcgaUserDrugCandidates", JSON.stringify(state.userDrugCandidates || []));
    }

    function loadUserDrugCandidates() {
      try {
        const rows = JSON.parse(localStorage.getItem("tcgaUserDrugCandidates") || "[]");
        state.userDrugCandidates = Array.isArray(rows) ? rows.filter(validUserDrugCandidate) : [];
      } catch {
        state.userDrugCandidates = [];
      }
    }

    function renderUserDrugList() {
      const node = document.getElementById("userDrugList");
      if (!node) return;
      const rows = state.userDrugCandidates || [];
      if (!rows.length) {
        node.innerHTML = `<div class="hint">${t("noUserDrugs")}</div>`;
        return;
      }
      node.innerHTML = rows.map((row, index) => `
        <div class="drug-user-row">
          <strong>${row.drug_name}</strong>
          <span>${t("targets")}: ${row.targets || row.primary_target} · ${t("validationScore")}: ${fmt(row.validation_score, 3)} · ${t("bbbEstimate")}: ${row.bbb_rule_label || "미확인"}</span>
          <button type="button" data-user-drug-index="${index}">${t("remove")}</button>
        </div>
      `).join("");
    }

    function tauLabel(value) {
      const tau = Number(value);
      if (!Number.isFinite(tau)) return "";
      return tau < 0 ? "reversal" : tau > 0 ? "mimicry" : "";
    }

    function selectedDrugRow(validationRows) {
      if (!validationRows.length) return null;
      return validationRows.find(row => row.drug_name === state.selectedDrugName) || validationRows[0];
    }

    function drugValidationScore(row) {
      const value = Number(row.validation_score ?? row.final_repurposing_score ?? row.repurposing_score ?? 0);
      return Number.isFinite(value) ? value : 0;
    }

    function rankedValidationRows(rows) {
      const byName = new Map();
      rows.forEach(row => {
        const key = canonicalDrugName(row.drug_name);
        if (!key) return;
        const existing = byName.get(key);
        if (!existing || drugValidationScore(row) >= drugValidationScore(existing)) {
          byName.set(key, row);
        }
      });
      return [...byName.values()]
        .sort((a, b) => drugValidationScore(b) - drugValidationScore(a))
        .map((row, index) => ({ ...row, validation_rank: index + 1 }));
    }

    function drugReasonItems(row) {
      const tau = Number(row.clue_tau_score);
      const targetDirection = degDirectionLabel(row.primary_target_direction);
      const targetReason = `${row.primary_target || ""} ${targetDirection ? `· ${targetDirection}` : ""} · log2FC ${fmt(row.primary_target_log2_fc, 2)} · FDR ${fmtSmall(row.primary_target_fdr, 2)}`;
      const clueReason = Number.isFinite(tau)
        ? `${t("clueTau")} ${fmt(tau, 2)} · ${tau < 0 ? "GBM 발현 signature를 반전시키는 방향" : "GBM 발현 signature와 유사한 방향"}`
        : "CLUE/LINCS tau score 미확인";
      const clinicalReason = [
        `${t("bbbEstimate")}: ${row.bbb_rule_label || "미확인"}`,
        `${t("gbmTrials")}: ${row.gbm_trial_count || 0}`,
        `${t("approved")}: ${yesNo(row.approved)}`,
        `${t("antineoplastic")}: ${yesNo(row.antineoplastic)}`
      ].join(" · ");
      return [
        [t("targetReason"), targetReason],
        [t("clueReason"), clueReason],
        [t("clinicalReason"), clinicalReason],
      ];
    }

    function renderSelectedDrug(row) {
      if (!row) {
        document.getElementById("drugSelectedDetail").innerHTML = "";
        return;
      }
      const tau = Number(row.clue_tau_score);
      const tauText = Number.isFinite(tau) ? `${fmt(tau, 2)} · ${tau < 0 ? "발현 반전" : "발현 유사"}` : "미확인";
      const reasonHtml = drugReasonItems(row).map(([title, body]) => `
        <div class="drug-reason-item">
          <strong>${title}</strong>
          <span>${body}</span>
        </div>
      `).join("");
      document.getElementById("drugSelectedDetail").innerHTML = `
        <div class="drug-selected-title">
          ${row.drug_name}
          <span>${row.validation_tier || ""}</span>
        </div>
        <p class="drug-selected-comment">${row.validation_comment_ko || ""}</p>
        <h3>${t("whySelected")}</h3>
        <div class="drug-reason-list">${reasonHtml}</div>
        <div class="drug-selected-grid">
          <div class="drug-selected-metric"><div>${t("validationScore")}</div><div>${fmt(row.validation_score, 3)}</div></div>
          <div class="drug-selected-metric"><div>${t("clueTau")}</div><div>${tauText}</div></div>
          <div class="drug-selected-metric"><div>${t("target")}</div><div>${row.primary_target || ""}</div></div>
          <div class="drug-selected-metric"><div>${t("bbbEstimate")}</div><div>${row.bbb_rule_label || ""}</div></div>
          <div class="drug-selected-metric"><div>${t("gbmTrials")}</div><div>${row.gbm_trial_count || 0}</div></div>
        </div>
      `;
    }

    function registeredPatientRows() {
      const rows = [...(state.wsiRiskPredictions || []), ...(state.predictions || [])];
      const seen = new Set();
      return rows.filter(row => {
        const caseId = row.case_submitter_id;
        if (!caseId || seen.has(caseId)) return false;
        seen.add(caseId);
        return true;
      });
    }

    function renderDrugPatientList() {
      const node = document.getElementById("drugPatientList");
      if (!node) return;
      const query = state.drugPatientSearch.trim().toLowerCase();
      const rows = registeredPatientRows()
        .filter(row => !query || String(row.case_submitter_id).toLowerCase().includes(query))
        .slice(0, 8);
      const selectedCaseId = activePatientCaseId();
      node.innerHTML = rows.map(row => `
        <button class="${row.case_submitter_id === selectedCaseId ? "is-selected" : ""}" type="button" data-case-id="${row.case_submitter_id}">
          <span>${row.case_submitter_id}</span>
          <span>${riskLabel(row.risk_group)}</span>
          <span>${fmt(row.risk_score, 2)}</span>
        </button>
      `).join("");
    }

    function setActivePatientFromRegisteredCase(caseId) {
      const prediction = registeredPatientRows().find(row => row.case_submitter_id === caseId);
      if (!prediction) return;
      const slide = matchingWsiForCase(caseId);
      setActivePatient({
        prediction,
        clinical: {
          case_submitter_id: caseId,
          project_id: prediction.project_id || slide?.project_id || "",
        },
        wsi: slide ? {
          name: slide.file_name || slide.file_id,
          file_name: slide.file_name,
          file_id: slide.file_id,
        } : null,
        gene_contributions: patientContribForCase(caseId, prediction.project_id || slide?.project_id || ""),
        clinical_contributions: [],
      });
    }

    function collectDrugCandidateForm() {
      const raw = {
        drug_name: document.getElementById("candidateDrugName").value,
        primary_target: document.getElementById("candidatePrimaryTarget").value,
        targets: document.getElementById("candidateTargets").value || document.getElementById("candidatePrimaryTarget").value,
        action: document.getElementById("candidateAction").value,
        validation_score: document.getElementById("candidateValidationScore").value,
        clue_tau_score: document.getElementById("candidateClueTau").value,
        bbb_rule_label: document.getElementById("candidateBbbLabel").value,
        approved: document.getElementById("candidateApproved").value,
        antineoplastic: document.getElementById("candidateAntineoplastic").value,
        gbm_trial_count: document.getElementById("candidateTrialCount").value,
        evidence_comment: document.getElementById("candidateEvidence").value,
      };
      return normalizeUserDrugCandidate(raw);
    }

    function upsertUserDrugCandidates(rows) {
      const existing = new Map((state.userDrugCandidates || []).map(row => [canonicalDrugName(row.drug_name), row]));
      rows.filter(validUserDrugCandidate).forEach(row => {
        existing.set(canonicalDrugName(row.drug_name), normalizeUserDrugCandidate(row));
      });
      state.userDrugCandidates = [...existing.values()].sort((a, b) => canonicalDrugName(a.drug_name).localeCompare(canonicalDrugName(b.drug_name)));
      saveUserDrugCandidates();
      renderUserDrugList();
      renderDrugRepurposing(state.drugs);
      renderPatientContexts();
    }

    function submitDrugCandidate(event) {
      event.preventDefault();
      const row = collectDrugCandidateForm();
      const status = document.getElementById("drugRegisterStatus");
      status.removeAttribute("data-i18n");
      if (!validUserDrugCandidate(row)) {
        status.textContent = t("drugMissingNameTarget");
        return;
      }
      upsertUserDrugCandidates([row]);
      status.textContent = t("drugAdded");
      document.getElementById("drugRegistrationForm").reset();
      document.getElementById("candidateValidationScore").value = "0.500";
      document.getElementById("candidateTrialCount").value = "0";
    }

    function fillDrugExample() {
      document.getElementById("candidateDrugName").value = "ExampleDrug-ABCC3";
      document.getElementById("candidatePrimaryTarget").value = "ABCC3";
      document.getElementById("candidateTargets").value = "ABCC3;TOP2A";
      document.getElementById("candidateAction").value = "inhibitor";
      document.getElementById("candidateValidationScore").value = "0.720";
      document.getElementById("candidateClueTau").value = "-82.5";
      document.getElementById("candidateBbbLabel").value = "BBB 물성 경계";
      document.getElementById("candidateApproved").value = "false";
      document.getElementById("candidateAntineoplastic").value = "false";
      document.getElementById("candidateTrialCount").value = "0";
      document.getElementById("candidateEvidence").value = "ABCC3 위험 기여 환자에서 우선 검토할 사용자 등록 후보입니다.";
    }

    async function previewDrugCandidateUpload() {
      const file = document.getElementById("userDrugFile").files[0];
      await previewDrugCandidateFile(file);
    }

    async function previewDrugCandidateFile(file) {
      const meta = document.getElementById("drugPreviewMeta");
      const table = document.getElementById("drugPreviewTable");
      const status = document.getElementById("drugPreviewStatus");
      state.drugPreviewRows = [];
      table.innerHTML = "";
      status.textContent = "";
      if (!file) {
        meta.setAttribute("data-i18n", "previewEmpty");
        meta.textContent = t("previewEmpty");
        return;
      }
      try {
        const formData = new FormData();
        formData.append("drug_file", file);
        const response = await fetch("/api/preview-drug-candidates", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Drug preview failed.");
        state.drugPreviewRows = (payload.rows || []).map(normalizeUserDrugCandidate).filter(validUserDrugCandidate);
        meta.removeAttribute("data-i18n");
        meta.textContent = `${payload.filename} · ${payload.row_count || payload.rows.length} rows · ${payload.columns.length} columns`;
        renderDrugPreviewTable(payload.preview || []);
      } catch (err) {
        meta.removeAttribute("data-i18n");
        meta.textContent = err.message;
      }
    }

    function renderDrugPreviewTable(rows) {
      const table = document.getElementById("drugPreviewTable");
      if (!rows.length) {
        table.innerHTML = "";
        return;
      }
      const columns = Object.keys(rows[0]).slice(0, 8);
      table.innerHTML = `
        <thead><tr>${columns.map(col => `<th>${col}</th>`).join("")}</tr></thead>
        <tbody>${rows.slice(0, 12).map(row => `
          <tr>${columns.map(col => `<td>${row[col] ?? ""}</td>`).join("")}</tr>
        `).join("")}</tbody>
      `;
    }

    async function applyDrugPreviewRows() {
      const status = document.getElementById("drugPreviewStatus");
      status.removeAttribute("data-i18n");
      if (!state.drugPreviewRows.length) {
        const file = document.getElementById("userDrugFile").files[0];
        if (file) await previewDrugCandidateFile(file);
      }
      if (!state.drugPreviewRows.length) {
        status.textContent = t("drugPreviewMissing");
        return;
      }
      upsertUserDrugCandidates(state.drugPreviewRows);
      status.textContent = `${state.drugPreviewRows.length} ${t("drugPreviewApplied")}`;
    }

    function removeUserDrugCandidate(index) {
      state.userDrugCandidates.splice(index, 1);
      saveUserDrugCandidates();
      renderUserDrugList();
      renderDrugRepurposing(state.drugs);
      renderPatientContexts();
    }

    function renderDrugRepurposing(payload) {
      const candidates = [...(payload?.candidates || []), ...(state.userDrugCandidates || [])];
      const shortlist = payload?.shortlist || [];
      const validationRows = rankedValidationRows([...(state.userDrugCandidates || []), ...(payload?.validation_matrix || [])]);
      const deg = payload?.deg || [];
      if (!state.selectedDrugName && validationRows.length) state.selectedDrugName = validationRows[0].drug_name;
      document.getElementById("finalDrugSummary").textContent = t("shortlistSummaryText")(payload?.final_summary || {});
      const overviewByName = new Map();
      [...shortlist, ...(state.userDrugCandidates || [])].forEach(row => {
        overviewByName.set(canonicalDrugName(row.drug_name), row);
      });
      const overviewRows = [...overviewByName.values()]
        .sort((a, b) => Number(b.final_repurposing_score ?? b.validation_score ?? 0) - Number(a.final_repurposing_score ?? a.validation_score ?? 0))
        .slice(0, Math.max(7, overviewByName.size));
      const maxOverviewScore = Math.max(...overviewRows.map(row => Number(row.final_repurposing_score || 0)), 0.001);
      document.getElementById("drugCandidateBars").innerHTML = overviewRows.map(row => {
        const score = Number(row.final_repurposing_score || 0);
        return `
          <div class="drug-rank-row">
            <div class="drug-rank-name" title="${row.drug_name}">${row.drug_name}</div>
            <div class="drug-rank-track" title="${t("clueTau")}: ${fmt(row.clue_tau_score, 2)}">
              <span class="drug-rank-fill" style="width:${Math.max(2, score / maxOverviewScore * 100)}%"></span>
            </div>
            <div class="drug-rank-score">${fmt(score, 3)}</div>
          </div>
        `;
      }).join("");
      document.getElementById("validationSummary").textContent = t("validationSummaryText")(payload?.validation_summary || {});
      document.getElementById("drugValidationTable").innerHTML = `
        <thead><tr><th>${t("validationRank")}</th><th>${t("drugName")}</th><th>${t("target")}</th><th>${t("validationTier")}</th><th>${t("validationScore")}</th><th>${t("clueTau")}</th><th>${t("bbbEstimate")}</th><th>${t("gbmTrials")}</th><th>${t("approved")}</th><th>${t("antineoplastic")}</th><th>${t("evidenceComment")}</th></tr></thead>
        <tbody>${validationRows.map(row => `
          <tr class="drug-row ${row.drug_name === state.selectedDrugName ? "is-selected" : ""}" data-drug-name="${row.drug_name}">
            <td>${row.validation_rank}</td>
            <td>${row.drug_name}</td>
            <td>${row.primary_target || ""}</td>
            <td>${row.validation_tier || ""}</td>
            <td><span class="score-chip">${fmt(row.validation_score, 3)}</span></td>
            <td><span class="tau-chip ${tauLabel(row.clue_tau_score)}">${fmt(row.clue_tau_score, 2)}</span></td>
            <td>${row.bbb_rule_label || ""} (${fmt(row.bbb_rule_score, 2)})</td>
            <td>${row.gbm_trial_count || 0}</td>
            <td>${yesNo(row.approved)}</td>
            <td>${yesNo(row.antineoplastic)}</td>
            <td>${row.validation_comment_ko || ""}</td>
          </tr>
        `).join("")}</tbody>
      `;
      document.getElementById("drugValidationTable").className = "drug-validation-table";
      renderSelectedDrug(selectedDrugRow(validationRows));
      document.getElementById("drugSummary").textContent = t("drugSummaryText")(payload?.summary || {});
      document.getElementById("clueSummary").textContent = t("clueSummaryText")(payload?.clue || {});
      document.getElementById("drugCandidateTable").innerHTML = `
        <thead><tr><th>${t("rank")}</th><th>${t("drugName")}</th><th>${t("finalRepurposingScore")}</th><th>${t("repurposingScore")}</th><th>${t("target")}</th><th>${t("direction")}</th><th>${t("reversalPrior")}</th><th>${t("clueStatus")}</th><th>${t("approved")}</th><th>${t("antineoplastic")}</th></tr></thead>
        <tbody>${candidates.map(row => `
          <tr>
            <td>${row.rank}</td>
            <td>${row.drug_name}</td>
            <td>${fmt(row.final_repurposing_score ?? row.repurposing_score, 3)}</td>
            <td>${fmt(row.repurposing_score, 3)}</td>
            <td>${row.primary_target}</td>
            <td>${degDirectionLabel(row.primary_target_direction)}</td>
            <td title="${row.local_reversal_reason || ""}">${fmt(row.local_reversal_prior, 2)}</td>
            <td>${row.clue_status || ""}</td>
            <td>${yesNo(row.approved)}</td>
            <td>${yesNo(row.antineoplastic)}</td>
          </tr>
        `).join("")}</tbody>
      `;
      document.getElementById("degTable").innerHTML = `
        <thead><tr><th>${t("gene")}</th><th>${t("direction")}</th><th>${t("log2fc")}</th><th>${t("fdr")}</th><th>${t("gbmExpression")}</th><th>${t("controlExpression")}</th></tr></thead>
        <tbody>${deg.map(row => `
          <tr>
            <td>${row.gene}</td>
            <td>${degDirectionLabel(row.direction)}</td>
            <td>${fmt(row.log2_fold_change, 2)}</td>
            <td>${fmtSmall(row.fdr_bh, 3)}</td>
            <td>${fmt(row.gbm_mean_tpm, 2)}</td>
            <td>${fmt(row.control_mean_tpm, 2)}</td>
          </tr>
        `).join("")}</tbody>
      `;
    }

    function renderContrib(rows) {
      const htmlRows = rows.slice(0, 18).map(r => `
        <tr>
          <td>${r.case_submitter_id}</td>
          <td>${r.rank}</td>
          <td>${r.gene}</td>
          <td>${directionLabel(r.direction)}</td>
          <td>${fmt(r.risk_contribution, 4)}</td>
        </tr>
      `).join("");
      document.getElementById("contribTable").innerHTML = `
        <thead><tr><th>${t("patient")}</th><th>${t("rank")}</th><th>${t("gene")}</th><th>${t("direction")}</th><th>${t("contribution")}</th></tr></thead>
        <tbody>${htmlRows}</tbody>
      `;
    }

    function wsiSlideByFileId(fileId) {
      return (state.wsi.slides || []).find(slide => slide.file_id === fileId);
    }

    function wsiFeatureForSlide(slide) {
      if (!slide) return null;
      return (state.wsiFeatures || []).find(row => row.file_id === slide.file_id)
        || (state.wsiFeatures || []).find(row => row.case_submitter_id === slide.case_submitter_id)
        || null;
    }

    function wsiPredictionForSlide(slide) {
      if (!slide) return null;
      return (state.wsiRiskPredictions || []).find(row => row.case_submitter_id === slide.case_submitter_id)
        || (state.predictions || []).find(row => row.case_submitter_id === slide.case_submitter_id)
        || null;
    }

    function wsiStatusLabel(value) {
      if (!value) return "";
      return value === "processed" ? t("processed") : value;
    }

    function selectWsiSlide(fileId) {
      const slide = wsiSlideByFileId(fileId);
      if (!slide) return;
      state.selectedWsiFileId = slide.file_id;
      state.selectedWsiPatchKey = null;
      state.selectedWsiPatchUrl = null;
      state.selectedWsiPatch = null;
      const select = document.getElementById("wsiSlide");
      if (select) select.value = slide.file_id;
      const prediction = wsiPredictionForSlide(slide);
      if (prediction) {
        state.activePatient = {
          prediction,
          clinical: {
            case_submitter_id: slide.case_submitter_id,
            project_id: slide.project_id,
          },
          wsi: {
            name: slide.file_name || slide.file_id,
            file_name: slide.file_name,
            file_id: slide.file_id,
          },
          gene_contributions: patientContribForCase(slide.case_submitter_id, prediction.project_id || slide.project_id),
          clinical_contributions: [],
        };
        renderPatientContexts();
        renderDrugPatientList();
      }
      renderWsi();
      if (state.wsiFeatures.length) renderWsiFeatures(state.wsiFeatures);
    }

    function selectWsiPatch(fileId, patchKey, patchUrl) {
      const slide = wsiSlideByFileId(fileId);
      if (!slide) return;
      state.selectedWsiFileId = slide.file_id;
      state.selectedWsiPatchKey = patchKey;
      state.selectedWsiPatchUrl = patchUrl;
      const select = document.getElementById("wsiSlide");
      if (select) select.value = slide.file_id;
      renderWsi();
      if (state.wsiFeatures.length) renderWsiFeatures(state.wsiFeatures);
    }

    function renderWsiSelectedDetails(slide) {
      const feature = wsiFeatureForSlide(slide);
      const prediction = wsiPredictionForSlide(slide);
      const details = [
        [t("risk"), prediction ? riskLabel(prediction.risk_group) : t("noPrediction"), "text"],
        [t("score"), prediction?.risk_score, "number"],
        [t("osDays"), prediction?.os_days, "integer"],
        [t("status"), prediction ? (Number(prediction.os_event) ? t("event") : t("censored")) : "", "text"],
        [t("patchesUsed"), feature?.wsi_patches_used ?? slide.patches],
        [t("tissueRatio"), feature?.wsi_tissue_fraction_thumbnail ?? slide.tissue_fraction_thumbnail],
        [t("tissueArea"), feature?.wsi_tissue_area_megapixels],
        [t("saturation"), feature?.wsi_saturation_mean_mean],
      ];
      const metrics = details.map(([label, value, kind]) => {
        const formatted = kind === "text"
          ? (value || "")
          : kind === "integer"
            ? (Number.isFinite(Number(value)) ? Math.round(Number(value)) : "")
            : (Number.isFinite(Number(value)) ? fmt(value, label === t("patchesUsed") ? 0 : 3) : "");
        return `
        <div class="wsi-detail-item">
          <div class="wsi-detail-label">${label}</div>
          <div class="wsi-detail-value">${formatted}</div>
        </div>
      `}).join("");
      document.getElementById("wsiSelectedDetails").innerHTML = `
        <div class="wsi-selected-head">
          <div>
            <div class="wsi-selected-title">${t("selectedWsiPatient")} · ${slide.case_submitter_id}</div>
            <div class="wsi-selected-subtitle">${t("survivalPrediction")} · ${slide.project_id} · ${t("slidePixels")} ${Math.round(Number(slide.width))} x ${Math.round(Number(slide.height))}</div>
          </div>
          <span class="pill">${wsiStatusLabel(slide.status)}</span>
        </div>
        <div class="wsi-detail-grid">${metrics}</div>
      `;
    }

    function renderSelectedPatchPreview(patch) {
      const enlargeButton = document.getElementById("enlargePatchButton");
      if (!patch) {
        state.selectedWsiPatch = null;
        if (enlargeButton) enlargeButton.disabled = true;
        document.getElementById("wsiPatchPreviewFrame").innerHTML = `<span class="hint">${t("noPatches")}</span>`;
        document.getElementById("wsiPatchPreviewMeta").innerHTML = "";
        return;
      }
      state.selectedWsiPatch = patch;
      if (enlargeButton) enlargeButton.disabled = false;
      state.selectedWsiPatchKey = `${patch.file_id}:${patch.rank}`;
      state.selectedWsiPatchUrl = patch.patch_image_url;
      const heatmapUrl = patch.patch_attention_heatmap_url;
      document.getElementById("wsiPatchPreviewFrame").innerHTML = heatmapUrl
        ? `
          <div class="patch-preview-pair">
            <div class="patch-preview-item">
              <img src="${patch.patch_image_url}" alt="${patch.case_submitter_id} patch ${patch.rank}">
              <span class="patch-preview-label">${t("originalPatch")}</span>
            </div>
            <div class="patch-preview-item">
              <img src="${heatmapUrl}" alt="${patch.case_submitter_id} patch ${patch.rank} attention heatmap">
              <span class="patch-preview-label">${t("attentionHeatmap")}</span>
            </div>
          </div>
        `
        : `<img src="${patch.patch_image_url}" alt="${patch.case_submitter_id} patch ${patch.rank}">`;
      document.getElementById("wsiPatchPreviewMeta").textContent = selectedPatchMetaText(patch);
    }

    function selectedPatchMetaText(patch) {
      return [
        `${t("patchRiskRank")}: ${patch.patch_risk_rank || patch.rank}`,
        `${patchDirectionLabel(patch.patch_risk_direction)}`,
        `${t("patchRiskScore")}: ${fmtSmall(patch.patch_risk_score, 3)}`,
        `${t("tissueRatio")}: ${fmt(patch.tissue_fraction, 3)}`
      ].filter(Boolean).join(" · ");
    }

    function openImageModalForSelectedPatch() {
      const patch = state.selectedWsiPatch;
      if (!patch) return;
      const heatmapUrl = patch.patch_attention_heatmap_url;
      document.getElementById("imageModalTitle").textContent =
        `${t("selectedPatch")} · ${patch.case_submitter_id} · R${patch.patch_risk_rank || patch.rank}`;
      document.getElementById("imageModalMeta").textContent = selectedPatchMetaText(patch);
      document.getElementById("imageModalBody").innerHTML = `
        <div class="image-modal-grid">
          <figure class="image-modal-figure">
            <img src="${patch.patch_image_url}" alt="${patch.case_submitter_id} patch ${patch.rank}">
            <figcaption class="image-modal-caption">${t("originalPatch")}</figcaption>
          </figure>
          ${heatmapUrl ? `
            <figure class="image-modal-figure">
              <img src="${heatmapUrl}" alt="${patch.case_submitter_id} patch ${patch.rank} attention heatmap">
              <figcaption class="image-modal-caption">${t("attentionHeatmap")}</figcaption>
            </figure>
          ` : ""}
        </div>
      `;
      const modal = document.getElementById("imageModal");
      modal.classList.add("active");
      modal.setAttribute("aria-hidden", "false");
    }

    function closeImageModal() {
      const modal = document.getElementById("imageModal");
      modal.classList.remove("active");
      modal.setAttribute("aria-hidden", "true");
    }

    function renderWsi() {
      const slides = state.wsi.slides || [];
      const patches = state.wsi.patches || [];
      if (!slides.length) {
        document.getElementById("wsiOverlayFrame").innerHTML = `<span class="hint">${t("noPatches")}</span>`;
        document.getElementById("wsiPatchPreviewFrame").innerHTML = `<span class="hint">${t("noPatches")}</span>`;
        document.getElementById("wsiPatchPreviewMeta").innerHTML = "";
        document.getElementById("wsiSelectedDetails").innerHTML = "";
        document.getElementById("wsiPatchGrid").innerHTML = "";
        return;
      }
      fillSelect("wsiSlide", slides.map(slide => [
        slide.file_id,
        `${slide.case_submitter_id} - ${slide.project_id} - ${slide.patches} ${t("patches")}`
      ]));
      if (!state.selectedWsiFileId || !wsiSlideByFileId(state.selectedWsiFileId)) {
        state.selectedWsiFileId = slides[0].file_id;
      }
      const select = document.getElementById("wsiSlide");
      select.value = state.selectedWsiFileId;
      const selectedId = state.selectedWsiFileId;
      const slide = slides.find(item => item.file_id === selectedId) || slides[0];
      if (!state.activePatient) {
        const prediction = wsiPredictionForSlide(slide);
        if (prediction) {
          state.activePatient = {
            prediction,
            clinical: {
              case_submitter_id: slide.case_submitter_id,
              project_id: slide.project_id,
            },
            wsi: {
              name: slide.file_name || slide.file_id,
              file_name: slide.file_name,
              file_id: slide.file_id,
            },
            gene_contributions: patientContribForCase(slide.case_submitter_id, prediction.project_id || slide.project_id),
            clinical_contributions: [],
          };
          renderPatientContexts();
        }
      }
      document.getElementById("wsiOverlayFrame").innerHTML = `<img src="${slide.risk_overlay_url || slide.overlay_url}" alt="${slide.case_submitter_id} WSI risk overlay">`;
      renderWsiSelectedDetails(slide);

      const slidePatches = patches.filter(patch => patch.file_id === slide.file_id)
        .sort((a, b) => {
          const ar = Number(a.patch_risk_rank);
          const br = Number(b.patch_risk_rank);
          if (Number.isFinite(ar) && Number.isFinite(br)) return ar - br;
          if (Number.isFinite(ar)) return -1;
          if (Number.isFinite(br)) return 1;
          return Number(a.rank) - Number(b.rank);
        })
        .slice(0, 8);
      const selectedPatch = slidePatches.find(patch => `${patch.file_id}:${patch.rank}` === state.selectedWsiPatchKey) || slidePatches[0];
      if (selectedPatch) {
        renderSelectedPatchPreview(selectedPatch);
      } else {
        renderSelectedPatchPreview(null);
      }
      document.getElementById("wsiPatchGrid").innerHTML = slidePatches.length
        ? slidePatches.map(patch => {
            const patchKey = `${patch.file_id}:${patch.rank}`;
            const selected = patchKey === state.selectedWsiPatchKey;
            const riskClass = patchRiskClass(patch.patch_risk_direction);
            const rankLabel = patch.patch_risk_rank ? `R${patch.patch_risk_rank}` : `#${patch.rank}`;
            const title = [
              patchDirectionLabel(patch.patch_risk_direction),
              `${t("patchRiskScore")}: ${fmtSmall(patch.patch_risk_score, 3)}`,
              `${t("tissueRatio")}: ${fmt(patch.tissue_fraction, 3)}`
            ].filter(Boolean).join(" · ");
            return `
              <button class="patch-tile ${riskClass} ${selected ? "is-selected" : ""}" type="button" data-file-id="${patch.file_id}" data-patch-key="${patchKey}" data-patch-url="${patch.patch_image_url}" title="${title}">
                <img src="${patch.patch_image_url}" alt="${patch.case_submitter_id} patch ${patch.rank}">
                <span class="patch-badge">${rankLabel}</span>
              </button>
            `;
          }).join("")
        : `<span class="hint">${t("noPatches")}</span>`;
    }

    function renderWsiFeatures(rows) {
      const slides = state.wsi.slides || [];
      const htmlRows = rows.map(row => {
        const slide = slides.find(item => item.file_id === row.file_id)
          || slides.find(item => item.case_submitter_id === row.case_submitter_id)
          || {};
        const fileId = row.file_id || slide.file_id || "";
        return `
        <tr class="wsi-row ${fileId === state.selectedWsiFileId ? "is-selected" : ""}" data-file-id="${fileId}" tabindex="0">
          <td class="col-patient" title="${row.case_submitter_id}">${row.case_submitter_id}</td>
          <td class="col-project">${row.project_id}</td>
          <td class="col-patches">${row.wsi_patches_used}</td>
          <td class="col-ratio">${fmt(row.wsi_tissue_fraction_thumbnail, 4)}</td>
          <td class="col-status">${wsiStatusLabel(slide.status)}</td>
          <td>${fmt(row.wsi_tissue_area_megapixels, 2)}</td>
          <td>${fmt(row.wsi_brightness_mean_mean, 3)}</td>
          <td>${fmt(row.wsi_saturation_mean_mean, 3)}</td>
          <td>${fmt(row.wsi_dark_pixel_ratio_mean, 3)}</td>
          <td>${fmt(row.wsi_purple_pixel_ratio_mean, 3)}</td>
          <td>${fmt(row.wsi_pink_pixel_ratio_mean, 3)}</td>
        </tr>
      `}).join("");
      document.getElementById("wsiFeatureTable").innerHTML = `
        <thead>
          <tr>
            <th class="col-patient">${t("patient")}</th>
            <th class="col-project">${t("project")}</th>
            <th class="col-patches">${t("patches")}</th>
            <th class="col-ratio">${t("tissueRatio")}</th>
            <th class="col-status">${t("status")}</th>
            <th>${t("tissueArea")}</th>
            <th>${t("brightness")}</th>
            <th>${t("saturation")}</th>
            <th>${t("darkRatio")}</th>
            <th>${t("purpleRatio")}</th>
            <th>${t("pinkRatio")}</th>
          </tr>
        </thead>
        <tbody>${htmlRows}</tbody>
      `;
      document.getElementById("wsiFeatureTable").className = "wsi-table";
    }

    function contributionTable(rows, featureLabel) {
      if (!rows || !rows.length) return `<p class="hint">${t("noPatches")}</p>`;
      const body = rows.slice(0, 10).map(row => `
        <tr>
          <td>${row.feature}</td>
          <td>${directionLabel(row.direction)}</td>
          <td>${fmt(row.contribution, 4)}</td>
        </tr>
      `).join("");
      return `
        <h2 style="margin-top:14px">${featureLabel}</h2>
        <div class="table-scroll" style="max-height:260px">
          <table>
            <thead><tr><th>${t("feature")}</th><th>${t("direction")}</th><th>${t("contribution")}</th></tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
    }

    function renderPredictionResult(payload) {
      const result = payload.prediction;
      const riskClass = result.risk_group === "High" ? "risk-high" : "risk-low";
      const container = document.getElementById("predictionResult");
      container.removeAttribute("data-i18n");
      container.className = "";
      container.innerHTML = `
        <div class="result-card">
          <div class="metric">
            <div class="label">${t("predictedRisk")}</div>
            <div class="value ${riskClass}">${riskLabel(result.risk_group)}</div>
            <div class="note">${result.case_submitter_id}</div>
          </div>
        </div>
        <table>
          <tbody>
            <tr><th>${t("riskScore")}</th><td>${fmt(result.risk_score, 4)}</td></tr>
            <tr><th>${t("trainingMedian")}</th><td>${fmt(result.training_median_risk, 4)}</td></tr>
          </tbody>
        </table>
        ${contributionTable(payload.gene_contributions, t("geneContributions"))}
        ${contributionTable(payload.clinical_contributions, t("clinicalContributions"))}
      `;
    }

    async function postPrediction(formData) {
      const response = await fetch("/api/predict-rna-clinical", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || t("predictionFailed"));
      return payload;
    }

    function buildClinicalRecord() {
      return {
        case_submitter_id: document.getElementById("caseSubmitterId").value.trim() || "uploaded_patient",
        project_id: document.getElementById("projectId").value,
        gender: document.getElementById("gender").value,
        age_at_diagnosis_years: Number(document.getElementById("ageAtDiagnosis").value || 0),
        tumor_grade: document.getElementById("tumorGrade").value.trim() || "Unknown",
        primary_diagnosis: document.getElementById("primaryDiagnosis").value.trim() || "Unknown",
        pancan_subtype: document.getElementById("pancanSubtype").value.trim() || "Unknown",
        cancer_type_detailed: document.getElementById("cancerTypeDetailed").value.trim() || "Unknown",
        tumor_type: document.getElementById("tumorType").value.trim() || "Unknown"
      };
    }

    function renderRnaPreview(payload) {
      const table = document.getElementById("rnaPreviewTable");
      const meta = document.getElementById("rnaPreviewMeta");
      meta.removeAttribute("data-i18n");
      meta.textContent = t("previewRows")(payload.rows, payload.columns.length);
      const columns = payload.columns.slice(0, 8);
      const header = columns.map(col => `<th>${col}</th>`).join("");
      const rows = (payload.preview || []).map(row => `
        <tr>${columns.map(col => `<td>${row[col] ?? ""}</td>`).join("")}</tr>
      `).join("");
      table.innerHTML = `<thead><tr>${header}</tr></thead><tbody>${rows}</tbody>`;
    }

    async function previewRnaUpload() {
      const file = document.getElementById("rnaFile").files[0];
      const table = document.getElementById("rnaPreviewTable");
      const meta = document.getElementById("rnaPreviewMeta");
      if (!file) {
        table.innerHTML = "";
        meta.textContent = t("previewEmpty");
        return;
      }
      const formData = new FormData();
      formData.append("rna_file", file);
      try {
        const response = await fetch("/api/preview-rna-upload", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || t("predictionFailed"));
        renderRnaPreview(payload);
      } catch (err) {
        table.innerHTML = "";
        meta.removeAttribute("data-i18n");
        meta.textContent = err.message;
      }
    }

    function updateWsiUploadPreview() {
      const file = document.getElementById("wsiImageFile").files[0];
      const preview = document.getElementById("wsiUploadPreview");
      if (state.pendingWsi?.url) URL.revokeObjectURL(state.pendingWsi.url);
      if (!file) {
        state.pendingWsi = null;
        preview.setAttribute("data-i18n", "wsiUploadEmpty");
        preview.textContent = t("wsiUploadEmpty");
        return;
      }
      const isPreviewable = file.type.startsWith("image/");
      state.pendingWsi = {
        name: file.name,
        size: file.size,
        type: file.type || "WSI",
        url: isPreviewable ? URL.createObjectURL(file) : null,
      };
      preview.removeAttribute("data-i18n");
      preview.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
      renderPatientContexts();
    }

    async function submitPrediction(event) {
      event.preventDefault();
      const rna = document.getElementById("rnaFile").files[0];
      const clinical = buildClinicalRecord();
      if (!rna || !clinical.case_submitter_id) {
        document.getElementById("predictionStatus").textContent = t("missingUpload");
        return;
      }
      const formData = new FormData();
      formData.append("rna_file", rna);
      formData.append(
        "clinical_json",
        new Blob([JSON.stringify(clinical, null, 2)], { type: "application/json" }),
        "clinical.json"
      );
      await runPrediction(formData);
    }

    async function runExamplePrediction() {
      await runPrediction(null);
    }

    async function runPrediction(formData) {
      const buttons = document.querySelectorAll("#predictionForm button");
      const status = document.getElementById("predictionStatus");
      buttons.forEach(button => button.disabled = true);
      status.textContent = t("predictionRunning");
      try {
        const payload = formData ? await postPrediction(formData) : await getJson("/api/predict-example");
        setActivePatientFromPrediction(payload, formData ? buildClinicalRecord() : null);
        renderPredictionResult(payload);
        status.textContent = "";
      } catch (err) {
        status.textContent = `${t("predictionFailed")}: ${err.message}`;
      } finally {
        buttons.forEach(button => button.disabled = false);
      }
    }

    async function refreshKm() {
      state.km = await getJson(`/api/km?run=${document.getElementById("kmRun").value}`);
      renderKm(state.km);
    }

    async function refreshPredictions() {
      state.predictions = await getJson(`/api/predictions?run=${document.getElementById("predRun").value}`);
      renderPredictions(state.predictions);
    }

    async function refreshContrib() {
      state.contrib = await getJson(`/api/patient-contrib?cohort=${document.getElementById("contribCohort").value}`);
      renderContrib(state.contrib);
    }

    function handleWsiRowClick(event) {
      const row = event.target.closest(".wsi-row[data-file-id]");
      if (row?.dataset?.fileId) selectWsiSlide(row.dataset.fileId);
    }

    function handleWsiRowKeydown(event) {
      if (event.key !== "Enter" && event.key !== " ") return;
      const row = event.target.closest(".wsi-row[data-file-id]");
      if (!row?.dataset?.fileId) return;
      event.preventDefault();
      selectWsiSlide(row.dataset.fileId);
    }

    function handleWsiPatchClick(event) {
      const tile = event.target.closest(".patch-tile[data-file-id]");
      if (!tile?.dataset?.fileId) return;
      selectWsiPatch(tile.dataset.fileId, tile.dataset.patchKey, tile.dataset.patchUrl);
    }

    function handlePerformanceCellClick(event) {
      const cell = event.target.closest(".performance-cell[data-performance-key]");
      if (cell?.dataset?.performanceKey) selectPerformance(cell.dataset.performanceKey);
    }

    function handleDrugRowClick(event) {
      const row = event.target.closest(".drug-row[data-drug-name]");
      if (!row?.dataset?.drugName) return;
      state.selectedDrugName = row.dataset.drugName;
      renderDrugRepurposing(state.drugs);
    }

    function handleDrugPatientClick(event) {
      const button = event.target.closest("button[data-case-id]");
      if (!button?.dataset?.caseId) return;
      setActivePatientFromRegisteredCase(button.dataset.caseId);
    }

    function rerenderAll() {
      applyStaticText();
      refreshSelectLabels();
      if (state.summary) renderStats(state.summary);
      if (state.comparison.length) {
        renderPerformance(state.comparison);
        renderComparisonTable(state.comparison);
      }
      if (state.genes.length) renderGenes(state.genes);
      if (state.km.length) renderKm(state.km);
      if (state.predictions.length) renderPredictions(state.predictions);
      if (state.contrib.length) renderContrib(state.contrib);
      if (state.drugs.candidates.length || state.drugs.deg.length) renderDrugRepurposing(state.drugs);
      if (state.wsi.slides.length) renderWsi();
      if (state.wsiFeatures.length) renderWsiFeatures(state.wsiFeatures);
      renderUserDrugList();
      renderPatientContexts();
      renderDrugPatientList();
    }

    async function init() {
      const savedLanguage = localStorage.getItem("tcgaDashboardLanguage") || "ko";
      document.getElementById("languageSelect").value = savedLanguage;
      loadUserDrugCandidates();
      fillSelect("kmRun", runOptions.map(key => [key, runLabel(key)]));
      fillSelect("predRun", runOptions.map(key => [key, runLabel(key)]));
      fillSelect("contribCohort", contribCohorts.map(([key, cohort]) => [key, cohortLabel(cohort)]));

      const [summary, comparison, genes, wsi, wsiFeatures, wsiRiskPredictions, drugs, contribAll, contribGbm] = await Promise.all([
        getJson("/api/summary"),
        getJson("/api/model-comparison"),
        getJson("/api/top-genes"),
        getJson("/api/wsi-smoke"),
        getJson("/api/wsi-features"),
        getJson("/api/predictions?run=all_rna_clinical&limit=0"),
        getJson("/api/drug-repurposing"),
        getJson("/api/patient-contrib?cohort=all"),
        getJson("/api/patient-contrib?cohort=gbm")
      ]);
      state.summary = summary;
      state.comparison = comparison;
      state.genes = genes;
      state.wsi = wsi;
      state.wsiFeatures = wsiFeatures;
      state.wsiRiskPredictions = wsiRiskPredictions;
      state.drugs = drugs;
      state.contribAll = contribAll;
      state.contribGbm = contribGbm;

      const cohorts = [...new Set(genes.map(r => r.cohort))];
      const analyses = [...new Set(genes.map(r => r.analysis))];
      fillSelect("geneCohort", cohorts.map(v => [v, cohortLabel(v)]));
      fillSelect("geneAnalysis", analyses.map(v => [v, analysisLabel(v)]));
      document.getElementById("geneAnalysis").value = "RNA adjusted for clinical";

      document.getElementById("languageSelect").addEventListener("change", () => {
        localStorage.setItem("tcgaDashboardLanguage", lang());
        rerenderAll();
      });
      document.querySelectorAll(".tab-button").forEach(button => {
        button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
      });
      document.querySelectorAll("[data-analysis-target]").forEach(button => {
        button.addEventListener("click", () => activateAnalysisTab(button.dataset.analysisTarget));
      });
      document.querySelectorAll("[data-risk-view-target]").forEach(button => {
        button.addEventListener("click", () => activateRiskView(button.dataset.riskViewTarget));
      });
      document.querySelectorAll("[data-drug-view-target]").forEach(button => {
        button.addEventListener("click", () => activateDrugView(button.dataset.drugViewTarget));
      });
      document.getElementById("geneCohort").addEventListener("change", () => renderGenes(state.genes));
      document.getElementById("geneAnalysis").addEventListener("change", () => renderGenes(state.genes));
      document.getElementById("kmRun").addEventListener("change", async () => {
        const run = document.getElementById("kmRun").value;
        document.getElementById("predRun").value = run;
        await refreshKm();
        await refreshPredictions();
      });
      document.getElementById("predRun").addEventListener("change", refreshPredictions);
      document.getElementById("contribCohort").addEventListener("change", refreshContrib);
      document.getElementById("wsiSlide").addEventListener("change", event => selectWsiSlide(event.target.value));
      document.getElementById("performanceMode").addEventListener("change", () => {
        state.selectedPerformanceKey = null;
        renderPerformance(state.comparison);
      });
      document.getElementById("performanceCells").addEventListener("click", handlePerformanceCellClick);
      document.getElementById("drugValidationTable").addEventListener("click", handleDrugRowClick);
      document.getElementById("drugRegistrationForm").addEventListener("submit", submitDrugCandidate);
      document.getElementById("drugExampleButton").addEventListener("click", fillDrugExample);
      document.getElementById("userDrugFile").addEventListener("change", previewDrugCandidateUpload);
      document.getElementById("drugUploadZone").addEventListener("dragover", event => {
        event.preventDefault();
        event.currentTarget.classList.add("drag-over");
      });
      document.getElementById("drugUploadZone").addEventListener("dragleave", event => {
        event.currentTarget.classList.remove("drag-over");
      });
      document.getElementById("drugUploadZone").addEventListener("drop", async event => {
        event.preventDefault();
        event.currentTarget.classList.remove("drag-over");
        const file = event.dataTransfer.files[0];
        if (!file) return;
        document.getElementById("userDrugFile").files = event.dataTransfer.files;
        await previewDrugCandidateFile(file);
      });
      document.getElementById("applyDrugPreviewButton").addEventListener("click", applyDrugPreviewRows);
      document.getElementById("userDrugList").addEventListener("click", event => {
        const button = event.target.closest("button[data-user-drug-index]");
        if (!button) return;
        removeUserDrugCandidate(Number(button.dataset.userDrugIndex));
      });
      document.getElementById("drugPatientList").addEventListener("click", handleDrugPatientClick);
      document.getElementById("drugPatientSearch").addEventListener("input", event => {
        state.drugPatientSearch = event.target.value;
        renderDrugPatientList();
      });
      document.getElementById("wsiFeatureTable").addEventListener("click", handleWsiRowClick);
      document.getElementById("wsiFeatureTable").addEventListener("keydown", handleWsiRowKeydown);
      document.getElementById("wsiPatchGrid").addEventListener("click", handleWsiPatchClick);
      document.getElementById("enlargePatchButton").addEventListener("click", openImageModalForSelectedPatch);
      document.getElementById("imageModalClose").addEventListener("click", closeImageModal);
      document.getElementById("imageModal").addEventListener("click", event => {
        if (event.target.id === "imageModal") closeImageModal();
      });
      document.addEventListener("keydown", event => {
        if (event.key === "Escape") closeImageModal();
      });
      document.getElementById("predictionForm").addEventListener("submit", submitPrediction);
      document.getElementById("rnaFile").addEventListener("change", previewRnaUpload);
      document.getElementById("wsiImageFile").addEventListener("change", updateWsiUploadPreview);
      document.getElementById("examplePredictButton").addEventListener("click", runExamplePrediction);

      rerenderAll();
      await refreshKm();
      await refreshPredictions();
      await refreshContrib();
    }

    init().catch(err => {
      document.body.innerHTML = `<main class="wrap"><div class="panel"><h1>${t("loadError")}</h1><p>${err.message}</p></div></main>`;
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_html(self):
        encoded = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_file(self, path, allowed_root):
        allowed_root = allowed_root.resolve()
        path = path.resolve()
        if allowed_root not in path.parents and path != allowed_root:
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path
        try:
            if path == "/":
                self.send_html()
            elif path == "/api/summary":
                self.send_json(data_summary())
            elif path == "/api/model-comparison":
                self.send_json(model_comparison())
            elif path == "/api/top-genes":
                self.send_json(top_genes())
            elif path == "/api/km":
                self.send_json(km_curve(query.get("run", ["all_rna_clinical"])[0]))
            elif path == "/api/predictions":
                raw_limit = query.get("limit", ["40"])[0]
                limit = 40
                try:
                    limit = int(raw_limit)
                except ValueError:
                    limit = 0 if raw_limit == "all" else 40
                self.send_json(predictions(query.get("run", ["all_rna_clinical"])[0], limit=limit))
            elif path == "/api/patient-contrib":
                self.send_json(patient_gene_contrib(query.get("cohort", ["all"])[0]))
            elif path == "/api/wsi-smoke":
                self.send_json(wsi_dataset(query.get("dataset", ["dev50"])[0]))
            elif path == "/api/wsi-features":
                self.send_json(wsi_features(query.get("dataset", ["dev50"])[0]))
            elif path == "/api/drug-repurposing":
                self.send_json(drug_repurposing())
            elif path == "/api/predict-example":
                self.send_json(example_prediction())
            elif path.startswith("/figures/"):
                name = unquote(path.removeprefix("/figures/"))
                self.send_file(RESULTS / "figures" / name, RESULTS / "figures")
            elif path == "/wsi-image":
                relative_path = query.get("path", [""])[0]
                if not relative_path:
                    self.send_error(400)
                    return
                self.send_file(ROOT / unquote(relative_path), ROOT / "data/processed")
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, explain=str(exc))

    def parse_multipart_uploads(self, content_type):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        if not message.is_multipart():
            raise ValueError("Expected multipart upload.")
        uploads = {}
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if "form-data" not in disposition:
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            if name and filename and payload:
                uploads[name] = {"filename": filename, "data": payload}
        return uploads

    def save_uploaded_file(self, uploads, name, path):
        upload = uploads.get(name)
        if not upload:
            raise ValueError(f"Missing upload: {name}.")
        path.write_bytes(upload["data"])

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/predict-rna-clinical", "/api/preview-rna-upload", "/api/preview-drug-candidates"}:
            self.send_error(404)
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_json({"error": "Expected multipart/form-data."}, status=400)
                return
            uploads = self.parse_multipart_uploads(content_type)
            if parsed.path == "/api/preview-rna-upload":
                rna_upload = uploads.get("rna_file")
                if not rna_upload:
                    self.send_json({"error": "Missing upload: rna_file."}, status=400)
                    return
                self.send_json(preview_rna_upload(rna_upload["data"], rna_upload["filename"]))
                return
            if parsed.path == "/api/preview-drug-candidates":
                drug_upload = uploads.get("drug_file")
                if not drug_upload:
                    self.send_json({"error": "Missing upload: drug_file."}, status=400)
                    return
                self.send_json(preview_drug_candidates_upload(drug_upload["data"], drug_upload["filename"]))
                return
            run_id = uuid.uuid4().hex[:12]
            out_dir = OUTPUTS / "predictions/dashboard_uploads" / run_id
            tmp_root = ROOT / "tmp/dashboard_uploads"
            tmp_root.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{run_id}_", dir=tmp_root) as tmp:
                tmp_dir = Path(tmp)
                rna_file = tmp_dir / "rna.tsv"
                clinical_json = tmp_dir / "clinical.json"
                rna_upload = uploads.get("rna_file")
                if not rna_upload:
                    raise ValueError("Missing upload: rna_file.")
                rna_frame = read_uploaded_table(rna_upload["data"], rna_upload["filename"])
                rna_frame.to_csv(rna_file, sep="\t", index=False)
                self.save_uploaded_file(uploads, "clinical_json", clinical_json)
                payload = run_rna_clinical_prediction(rna_file, clinical_json, out_dir)
            payload["prediction"]["run_id"] = run_id
            self.send_json(payload)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--check", action="store_true", help="Validate dashboard data without starting a web server.")
    args = parser.parse_args()

    if args.check:
        summary = data_summary()
        comparison = model_comparison()
        genes = top_genes()
        km = km_curve("all_rna_clinical")
        pred = predictions("all_rna_clinical")
        wsi = wsi_dataset()
        wsi_feature_rows = wsi_features()
        wsi_patch_importance = read_tsv("results/wsi_uni_patch_importance.tsv")
        wsi_patch_heatmaps = read_tsv("results/wsi_uni_patch_heatmaps.tsv")
        drug = drug_repurposing()
        checks = {
            "patients": summary["patients"],
            "model_rows": len(comparison),
            "top_gene_rows": len(genes),
            "km_rows": len(km),
            "prediction_rows": len(pred),
            "wsi_dataset": wsi["dataset"],
            "wsi_slides": len(wsi["slides"]),
            "wsi_patch_images": len(wsi["patches"]),
            "wsi_patch_importance_rows": len(wsi_patch_importance),
            "wsi_patch_heatmap_rows": len(wsi_patch_heatmaps),
            "wsi_feature_rows": len(wsi_feature_rows),
            "drug_candidate_rows": len(drug["candidates"]),
            "drug_shortlist_rows": len(drug["shortlist"]),
            "drug_validation_rows": len(drug["validation_matrix"]),
            "deg_rows": len(drug["deg"]),
            "clue_status": drug["clue"].get("clue_status"),
            "clue_up_genes": drug["clue"].get("up_genes"),
            "clue_down_genes": drug["clue"].get("down_genes"),
            "final_model_exists": (MODEL_DIR / "model.joblib").exists(),
            "rna_matrix_exists": summary["rna_matrix_exists"],
        }
        print(json.dumps(checks, indent=2))
        if not all([checks["patients"], checks["model_rows"], checks["top_gene_rows"], checks["km_rows"], checks["prediction_rows"], checks["wsi_feature_rows"], checks["final_model_exists"], checks["rna_matrix_exists"]]):
            raise SystemExit(1)
        return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
