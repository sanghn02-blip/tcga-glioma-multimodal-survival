#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import pymysql

from db_config import mysql_config


ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"nan", "none", "null"}:
            return None
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def number(value, integer=False):
    value = clean(value)
    if value is None:
        return None
    try:
        converted = float(value)
    except ValueError:
        return None
    return int(converted) if integer else converted


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_key_value_tsv(path):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    return rows


def upsert_many(cursor, table, columns, rows, update_columns=None):
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    quoted = ", ".join(f"`{column}`" for column in columns)
    update_columns = update_columns if update_columns is not None else columns
    updates = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in update_columns)
    sql = f"INSERT INTO `{table}` ({quoted}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
    values = [[row.get(column) for column in columns] for row in rows]
    cursor.executemany(sql, values)
    return len(rows)


def load_patients(cursor, path):
    rows = []
    for row in read_tsv(path):
        rows.append(
            {
                "case_submitter_id": clean(row.get("case_submitter_id")),
                "project_id": clean(row.get("project_id")),
                "os_event": number(row.get("os_event"), integer=True),
                "os_days": number(row.get("os_days")),
                "vital_status": clean(row.get("vital_status")),
                "gender": clean(row.get("gender")),
                "age_at_diagnosis_days": number(row.get("age_at_diagnosis_days")),
                "tumor_grade": clean(row.get("tumor_grade")),
                "primary_diagnosis": clean(row.get("primary_diagnosis")),
                "wsi_file_count": number(row.get("wsi_file_count"), integer=True) or 0,
                "rna_file_count": number(row.get("rna_file_count"), integer=True) or 0,
            }
        )
    columns = [
        "case_submitter_id",
        "project_id",
        "os_event",
        "os_days",
        "vital_status",
        "gender",
        "age_at_diagnosis_days",
        "tumor_grade",
        "primary_diagnosis",
        "wsi_file_count",
        "rna_file_count",
    ]
    return upsert_many(cursor, "patients", columns, rows)


def local_file_path(root, file_id, file_name):
    candidates = [root / file_id / file_name, root / file_name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = list(root.rglob(file_name)) if root.exists() else []
    return str(matches[0]) if matches else None


def load_gdc_files(cursor, path, download_root=None):
    rows = []
    for row in read_tsv(path):
        file_id = clean(row.get("file_id"))
        file_name = clean(row.get("file_name"))
        rows.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "case_submitter_id": clean(row.get("case_submitter_id")),
                "project_id": clean(row.get("project_id")),
                "sample_submitter_ids": clean(row.get("sample_submitter_ids")),
                "data_type": clean(row.get("data_type")),
                "experimental_strategy": clean(row.get("experimental_strategy")),
                "workflow_type": clean(row.get("workflow_type")),
                "file_size": number(row.get("file_size"), integer=True),
                "local_path": local_file_path(download_root, file_id, file_name) if download_root else None,
            }
        )
    columns = [
        "file_id",
        "file_name",
        "case_submitter_id",
        "project_id",
        "sample_submitter_ids",
        "data_type",
        "experimental_strategy",
        "workflow_type",
        "file_size",
        "local_path",
    ]
    return upsert_many(cursor, "gdc_files", columns, rows)


def load_cbio(cursor, path):
    cursor.execute("SELECT case_submitter_id FROM patients")
    eligible_cases = {row[0] for row in cursor.fetchall()}
    rows = []
    for row in read_tsv(path):
        case_id = clean(row.get("case_submitter_id"))
        if case_id not in eligible_cases:
            continue
        rows.append({key: clean(row.get(key)) for key in [
            "case_submitter_id",
            "cbioportal_study_id",
            "pancan_subtype",
            "histologic_grade",
            "cancer_type_detailed",
            "tumor_type",
        ]})
    columns = [
        "case_submitter_id",
        "cbioportal_study_id",
        "pancan_subtype",
        "histologic_grade",
        "cancer_type_detailed",
        "tumor_type",
    ]
    return upsert_many(cursor, "cbioportal_covariates", columns, rows)


def load_artifacts(cursor):
    artifacts = [
        ("rna_matrix", "dev_20_tpm", "data/processed/rna_tpm_dev_20.tsv", "20-patient dev RNA TPM matrix"),
        ("predictions", "rna_dev_baseline_predictions", "results/rna_dev_baseline/rna_cox_predictions.tsv", "RNA dev Cox predictions"),
        ("metrics", "rna_dev_baseline_metrics", "results/rna_dev_baseline/rna_cox_metrics.tsv", "RNA dev Cox metrics"),
        ("km_curve", "rna_dev_baseline_km", "results/rna_dev_baseline/rna_cox_km_curve.tsv", "RNA dev Kaplan-Meier curve table"),
    ]
    rows = [
        {
            "artifact_type": artifact_type,
            "name": name,
            "path": path,
            "description": description,
        }
        for artifact_type, name, path, description in artifacts
        if (ROOT / path).exists()
    ]
    return upsert_many(cursor, "artifacts", ["artifact_type", "name", "path", "description"], rows, ["path", "description"])


def load_model_run(cursor, run_id, results_dir):
    metrics_path = results_dir / "rna_cox_metrics.tsv"
    predictions_path = results_dir / "rna_cox_predictions.tsv"
    if not metrics_path.exists() or not predictions_path.exists():
        return {"model_runs": 0, "model_metrics": 0, "model_predictions": 0}

    upsert_many(
        cursor,
        "model_runs",
        ["run_id", "model_name", "modality", "cohort_name", "notes"],
        [
            {
                "run_id": run_id,
                "model_name": "elastic_net_cox",
                "modality": "rna",
                "cohort_name": "dev_20",
                "notes": "20-patient RNA dev smoke test; not for scientific interpretation.",
            }
        ],
        ["model_name", "modality", "cohort_name", "notes"],
    )

    metric_rows = []
    for name, value in read_key_value_tsv(metrics_path):
        metric_rows.append(
            {
                "run_id": run_id,
                "metric_name": clean(name),
                "metric_value": number(value),
                "metric_text": clean(value),
            }
        )
    metric_count = upsert_many(
        cursor,
        "model_metrics",
        ["run_id", "metric_name", "metric_value", "metric_text"],
        metric_rows,
        ["metric_value", "metric_text"],
    )

    prediction_rows = []
    for row in read_tsv(predictions_path):
        prediction_rows.append(
            {
                "run_id": run_id,
                "case_submitter_id": clean(row.get("case_submitter_id")),
                "fold": number(row.get("fold"), integer=True),
                "risk_score": number(row.get("risk_score")),
                "os_days": number(row.get("os_days")),
                "os_event": number(row.get("os_event"), integer=True),
                "project_id": clean(row.get("project_id")),
                "risk_group": clean(row.get("risk_group")),
            }
        )
    prediction_count = upsert_many(
        cursor,
        "model_predictions",
        ["run_id", "case_submitter_id", "fold", "risk_score", "os_days", "os_event", "project_id", "risk_group"],
        prediction_rows,
        ["fold", "risk_score", "os_days", "os_event", "project_id", "risk_group"],
    )
    return {"model_runs": 1, "model_metrics": metric_count, "model_predictions": prediction_count}


def summarize(cursor):
    queries = {
        "patients": "SELECT COUNT(*) FROM patients",
        "gdc_files": "SELECT COUNT(*) FROM gdc_files",
        "rna_files": "SELECT COUNT(*) FROM gdc_files WHERE data_type = 'Gene Expression Quantification'",
        "wsi_files": "SELECT COUNT(*) FROM gdc_files WHERE data_type = 'Slide Image'",
        "cbioportal_covariates": "SELECT COUNT(*) FROM cbioportal_covariates",
        "model_runs": "SELECT COUNT(*) FROM model_runs",
        "model_predictions": "SELECT COUNT(*) FROM model_predictions",
        "artifacts": "SELECT COUNT(*) FROM artifacts",
    }
    for label, query in queries.items():
        cursor.execute(query)
        print(f"{label}: {cursor.fetchone()[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=Path("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv"))
    parser.add_argument("--rna-files", type=Path, default=Path("outputs/gdc_rna_file_map_primary_overlap.tsv"))
    parser.add_argument("--wsi-files", type=Path, default=Path("outputs/gdc_wsi_file_map_primary_overlap.tsv"))
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument("--rna-download-root", type=Path, default=Path("data/gdc/rna"))
    parser.add_argument("--wsi-download-root", type=Path, default=Path("data/gdc/wsi"))
    parser.add_argument("--include-dev-results", action="store_true")
    args = parser.parse_args()

    connection = pymysql.connect(**mysql_config())
    counts = {}
    try:
        with connection.cursor() as cursor:
            counts["patients"] = load_patients(cursor, args.cohort)
            counts["rna_files"] = load_gdc_files(cursor, args.rna_files, args.rna_download_root)
            counts["wsi_files"] = load_gdc_files(cursor, args.wsi_files, args.wsi_download_root)
            counts["cbioportal_covariates"] = load_cbio(cursor, args.cbio)
            counts["artifacts"] = load_artifacts(cursor)
            if args.include_dev_results:
                counts.update(load_model_run(cursor, "rna_dev_cox_v1", ROOT / "results/rna_dev_baseline"))
            connection.commit()
            for label, count in counts.items():
                print(f"loaded {label}: {count}")
            summarize(cursor)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
