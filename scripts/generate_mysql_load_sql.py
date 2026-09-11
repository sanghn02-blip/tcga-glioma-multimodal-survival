#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
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
    if math.isnan(converted):
        return None
    return int(converted) if integer else converted


def boolean_number(value):
    value = clean(value)
    if value is None:
        return None
    return 1 if value.lower() in {"1", "true", "yes", "y"} else 0


def sql_value(value):
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


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


def insert_statement(table, columns, row, update_columns=None):
    quoted = ", ".join(f"`{column}`" for column in columns)
    values = ", ".join(sql_value(row.get(column)) for column in columns)
    update_columns = update_columns if update_columns is not None else columns
    updates = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in update_columns)
    return f"INSERT INTO `{table}` ({quoted}) VALUES ({values}) ON DUPLICATE KEY UPDATE {updates};"


def local_file_path(root, file_id, file_name):
    candidates = [root / file_id / file_name, root / file_name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = list(root.rglob(file_name)) if root.exists() else []
    return str(matches[0]) if matches else None


def patient_rows(path):
    for row in read_tsv(path):
        yield {
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


def gdc_file_rows(path, download_root):
    for row in read_tsv(path):
        file_id = clean(row.get("file_id"))
        file_name = clean(row.get("file_name"))
        yield {
            "file_id": file_id,
            "file_name": file_name,
            "case_submitter_id": clean(row.get("case_submitter_id")),
            "project_id": clean(row.get("project_id")),
            "sample_submitter_ids": clean(row.get("sample_submitter_ids")),
            "data_type": clean(row.get("data_type")),
            "experimental_strategy": clean(row.get("experimental_strategy")),
            "workflow_type": clean(row.get("workflow_type")),
            "file_size": number(row.get("file_size"), integer=True),
            "local_path": local_file_path(download_root, file_id, file_name),
        }


def cbio_rows(path, eligible_cases):
    for row in read_tsv(path):
        case_id = clean(row.get("case_submitter_id"))
        if case_id not in eligible_cases:
            continue
        yield {
            "case_submitter_id": case_id,
            "cbioportal_study_id": clean(row.get("cbioportal_study_id")),
            "pancan_subtype": clean(row.get("pancan_subtype")),
            "histologic_grade": clean(row.get("histologic_grade")),
            "cancer_type_detailed": clean(row.get("cancer_type_detailed")),
            "tumor_type": clean(row.get("tumor_type")),
        }


def artifact_rows():
    artifacts = [
        ("rna_matrix", "dev_20_tpm", "data/processed/rna_tpm_dev_20.tsv", "20-patient dev RNA TPM matrix"),
        ("predictions", "rna_dev_baseline_predictions", "results/rna_dev_baseline/rna_cox_predictions.tsv", "RNA dev Cox predictions"),
        ("metrics", "rna_dev_baseline_metrics", "results/rna_dev_baseline/rna_cox_metrics.tsv", "RNA dev Cox metrics"),
        ("km_curve", "rna_dev_baseline_km", "results/rna_dev_baseline/rna_cox_km_curve.tsv", "RNA dev Kaplan-Meier curve table"),
        ("deg", "gbm_deg_proxy", "results/gbm_deg_proxy.tsv", "GBM-vs-LGG proxy differential expression table"),
        (
            "drug_repurposing",
            "gbm_drug_repurposing_candidates",
            "results/drug_repurposing_candidates.tsv",
            "DGIdb target-based drug repurposing candidate ranking",
        ),
        (
            "drug_repurposing",
            "dgidb_gbm_target_interactions",
            "results/dgidb_gbm_target_interactions.tsv",
            "DGIdb interactions for prioritized GBM differential-expression targets",
        ),
        (
            "xena_expression",
            "xena_gbm_gtex_brain_targeted_expression",
            "data/processed/xena_gbm_gtex_brain_targeted_expression.tsv",
            "Targeted UCSC Xena Toil expression matrix for TCGA GBM and GTEx normal brain samples",
        ),
        (
            "deg",
            "gbm_vs_gtex_brain_deg_targeted",
            "results/gbm_vs_gtex_brain_deg_targeted.tsv",
            "Targeted TCGA GBM primary tumor vs GTEx normal brain differential expression table",
        ),
        (
            "drug_repurposing",
            "gbm_gtex_brain_drug_repurposing_candidates",
            "results/drug_repurposing_candidates_gtex_brain.tsv",
            "DGIdb target-based drug repurposing candidate ranking using GTEx normal brain comparison",
        ),
        (
            "drug_repurposing",
            "dgidb_gbm_normal_targeted_interactions",
            "results/dgidb_gbm_normal_targeted_interactions.tsv",
            "DGIdb interactions for targeted GBM-vs-normal-brain differential-expression genes",
        ),
        (
            "clue_lincs",
            "gbm_vs_gtex_brain_clue_up_gmt",
            "results/clue_lincs/gbm_vs_gtex_brain_up.gmt",
            "CLUE/LINCS query input: genes higher in TCGA GBM than GTEx normal brain",
        ),
        (
            "clue_lincs",
            "gbm_vs_gtex_brain_clue_down_gmt",
            "results/clue_lincs/gbm_vs_gtex_brain_down.gmt",
            "CLUE/LINCS query input: genes lower in TCGA GBM than GTEx normal brain",
        ),
        (
            "clue_lincs",
            "gbm_clue_ready_drug_candidates",
            "results/drug_repurposing_candidates_clue_ready.tsv",
            "Drug repurposing candidate ranking with local CLUE/LINCS reversal-prior fields",
        ),
        (
            "clue_lincs",
            "gbm_drug_candidates_with_clue_tau",
            "results/drug_repurposing_candidates_with_clue_tau.tsv",
            "Drug repurposing candidate ranking with integrated CLUE/LINCS tau scores",
        ),
        (
            "clue_lincs",
            "clue_candidate_pert_metadata",
            "results/clue_lincs/clue_candidate_pert_metadata.tsv",
            "CLUE perturbagen metadata used to map BRD perturbagen IDs to candidate drug names",
        ),
        (
            "drug_repurposing",
            "gbm_drug_repurposing_final_shortlist",
            "results/drug_repurposing_final_shortlist.tsv",
            "Final research-only GBM drug repurposing shortlist with mechanism rationale and validation status",
        ),
        (
            "drug_repurposing",
            "gbm_drug_repurposing_validation_matrix",
            "results/drug_repurposing_validation_matrix.tsv",
            "Candidate validation matrix with PubChem physicochemical filters and ClinicalTrials.gov GBM trial flags",
        ),
    ]
    for artifact_type, name, path, description in artifacts:
        if (ROOT / path).exists():
            yield {
                "artifact_type": artifact_type,
                "name": name,
                "path": path,
                "description": description,
            }


def model_run_rows(include_dev_results):
    if not include_dev_results:
        return
    results_dir = ROOT / "results/rna_dev_baseline"
    if (results_dir / "rna_cox_metrics.tsv").exists() and (results_dir / "rna_cox_predictions.tsv").exists():
        yield {
            "run_id": "rna_dev_cox_v1",
            "model_name": "elastic_net_cox",
            "modality": "rna",
            "cohort_name": "dev_20",
            "notes": "20-patient RNA dev smoke test; not for scientific interpretation.",
        }


def metric_rows(include_dev_results):
    if not include_dev_results:
        return
    metrics_path = ROOT / "results/rna_dev_baseline/rna_cox_metrics.tsv"
    if not metrics_path.exists():
        return
    for name, value in read_key_value_tsv(metrics_path):
        yield {
            "run_id": "rna_dev_cox_v1",
            "metric_name": clean(name),
            "metric_value": number(value),
            "metric_text": clean(value),
        }


def prediction_rows(include_dev_results):
    if not include_dev_results:
        return
    predictions_path = ROOT / "results/rna_dev_baseline/rna_cox_predictions.tsv"
    if not predictions_path.exists():
        return
    for row in read_tsv(predictions_path):
        yield {
            "run_id": "rna_dev_cox_v1",
            "case_submitter_id": clean(row.get("case_submitter_id")),
            "fold": number(row.get("fold"), integer=True),
            "risk_score": number(row.get("risk_score")),
            "os_days": number(row.get("os_days")),
            "os_event": number(row.get("os_event"), integer=True),
            "project_id": clean(row.get("project_id")),
            "risk_group": clean(row.get("risk_group")),
        }


def deg_rows(path):
    if not path.exists():
        return
    for row in read_tsv(path):
        yield {
            "comparison_name": clean(row.get("comparison")) or "TCGA-GBM vs TCGA-LGG proxy",
            "gene": clean(row.get("gene")),
            "direction": clean(row.get("direction")),
            "log2_fold_change": number(row.get("log2_fold_change")),
            "fdr_bh": number(row.get("fdr_bh")),
            "gbm_mean_tpm": number(row.get("gbm_mean_tpm")),
            "control_mean_tpm": number(row.get("control_mean_tpm")),
            "survival_risk_direction": clean(row.get("survival_risk_direction")),
            "survival_coef": number(row.get("survival_coef")),
        }


def drug_candidate_rows(path):
    if not path.exists():
        return
    for row in read_tsv(path):
        yield {
            "comparison_name": clean(row.get("comparison")) or "TCGA-GBM vs TCGA-LGG proxy",
            "candidate_rank": number(row.get("rank"), integer=True),
            "drug_name": clean(row.get("drug_name")),
            "repurposing_score": number(row.get("repurposing_score")),
            "primary_target": clean(row.get("primary_target")),
            "targets": clean(row.get("targets")),
            "target_count": number(row.get("target_count"), integer=True),
            "primary_target_direction": clean(row.get("primary_target_direction")),
            "primary_target_log2_fc": number(row.get("primary_target_log2_fc")),
            "primary_target_fdr": number(row.get("primary_target_fdr")),
            "interaction_types": clean(row.get("interaction_types")),
            "sources": clean(row.get("sources")),
            "approved": boolean_number(row.get("approved")),
            "antineoplastic": boolean_number(row.get("antineoplastic")),
            "immunotherapy": boolean_number(row.get("immunotherapy")),
            "ranking_note": clean(row.get("ranking_note")),
        }


def clue_candidate_rows(path):
    if not path.exists():
        return
    for row in read_tsv(path):
        yield {
            "comparison_name": clean(row.get("comparison")) or "TCGA-GBM primary tumor vs GTEx normal brain",
            "candidate_rank": number(row.get("rank"), integer=True),
            "drug_name": clean(row.get("drug_name")),
            "final_repurposing_score": number(row.get("final_repurposing_score")),
            "repurposing_score": number(row.get("repurposing_score")),
            "primary_target": clean(row.get("primary_target")),
            "primary_target_direction": clean(row.get("primary_target_direction")),
            "local_reversal_prior": number(row.get("local_reversal_prior")),
            "local_reversal_reason": clean(row.get("local_reversal_reason")),
            "clue_tau_score": number(row.get("clue_tau_score")),
            "clue_connectivity_direction": clean(row.get("clue_connectivity_direction")),
            "clue_status": clean(row.get("clue_status")),
            "approved": boolean_number(row.get("approved")),
            "antineoplastic": boolean_number(row.get("antineoplastic")),
            "ranking_note": clean(row.get("ranking_note")),
        }


def shortlist_rows(path):
    if not path.exists():
        return
    for row in read_tsv(path):
        yield {
            "comparison_name": "TCGA-GBM primary tumor vs GTEx normal brain",
            "shortlist_rank": number(row.get("shortlist_rank"), integer=True),
            "candidate_tier": clean(row.get("candidate_tier")),
            "drug_name": clean(row.get("drug_name")),
            "primary_target": clean(row.get("primary_target")),
            "final_repurposing_score": number(row.get("final_repurposing_score")),
            "repurposing_score": number(row.get("repurposing_score")),
            "primary_target_direction": clean(row.get("primary_target_direction")),
            "primary_target_log2_fc": number(row.get("primary_target_log2_fc")),
            "primary_target_fdr": number(row.get("primary_target_fdr")),
            "local_reversal_prior": number(row.get("local_reversal_prior")),
            "mechanism_summary_ko": clean(row.get("mechanism_summary_ko")),
            "approved": boolean_number(row.get("approved")),
            "antineoplastic": boolean_number(row.get("antineoplastic")),
            "clue_status": clean(row.get("clue_status")),
            "clue_tau_score": number(row.get("clue_tau_score")),
            "clue_connectivity_direction": clean(row.get("clue_connectivity_direction")),
            "pert_id": clean(row.get("pert_id")),
            "clue_pert_iname": clean(row.get("clue_pert_iname")),
            "validation_status_ko": clean(row.get("validation_status_ko")),
            "rationale_ko": clean(row.get("rationale_ko")),
            "next_validation_ko": clean(row.get("next_validation_ko")),
        }


def validation_matrix_rows(path):
    if not path.exists():
        return
    for row in read_tsv(path):
        yield {
            "comparison_name": "TCGA-GBM primary tumor vs GTEx normal brain",
            "validation_rank": number(row.get("validation_rank"), integer=True),
            "validation_tier": clean(row.get("validation_tier")),
            "drug_name": clean(row.get("drug_name")),
            "primary_target": clean(row.get("primary_target")),
            "validation_score": number(row.get("validation_score")),
            "final_repurposing_score": number(row.get("final_repurposing_score")),
            "bbb_rule_score": number(row.get("bbb_rule_score")),
            "bbb_rule_label": clean(row.get("bbb_rule_label")),
            "molecular_weight": number(row.get("molecular_weight")),
            "xlogp": number(row.get("xlogp")),
            "tpsa": number(row.get("tpsa")),
            "pert_id": clean(row.get("pert_id")),
            "clue_pert_iname": clean(row.get("clue_pert_iname")),
            "clue_tau_score": number(row.get("clue_tau_score")),
            "gbm_trial_count": number(row.get("gbm_trial_count"), integer=True),
            "gbm_trial_nct_ids": clean(row.get("gbm_trial_nct_ids")),
            "external_lookup_status": clean(row.get("external_lookup_status")),
            "validation_comment_ko": clean(row.get("validation_comment_ko")),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=Path("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv"))
    parser.add_argument("--rna-files", type=Path, default=Path("outputs/gdc_rna_file_map_primary_overlap.tsv"))
    parser.add_argument("--wsi-files", type=Path, default=Path("outputs/gdc_wsi_file_map_primary_overlap.tsv"))
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument("--rna-download-root", type=Path, default=Path("data/gdc/rna"))
    parser.add_argument("--wsi-download-root", type=Path, default=Path("data/gdc/wsi"))
    parser.add_argument("--include-dev-results", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("db/load_tcga.sql"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    patients = list(patient_rows(args.cohort))
    eligible_cases = {row["case_submitter_id"] for row in patients}
    clue_candidates_path = ROOT / "results/drug_repurposing_candidates_with_clue_tau.tsv"
    if not clue_candidates_path.exists():
        clue_candidates_path = ROOT / "results/drug_repurposing_candidates_clue_ready.tsv"
    validation_matrix_path = ROOT / "results/drug_repurposing_validation_matrix_with_clue_tau.tsv"
    if not validation_matrix_path.exists():
        validation_matrix_path = ROOT / "results/drug_repurposing_validation_matrix.tsv"

    tables = [
        (
            "patients",
            [
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
            ],
            patients,
            None,
        ),
        (
            "gdc_files",
            [
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
            ],
            list(gdc_file_rows(args.rna_files, args.rna_download_root)) + list(gdc_file_rows(args.wsi_files, args.wsi_download_root)),
            None,
        ),
        (
            "cbioportal_covariates",
            [
                "case_submitter_id",
                "cbioportal_study_id",
                "pancan_subtype",
                "histologic_grade",
                "cancer_type_detailed",
                "tumor_type",
            ],
            list(cbio_rows(args.cbio, eligible_cases)),
            None,
        ),
        (
            "artifacts",
            ["artifact_type", "name", "path", "description"],
            list(artifact_rows()),
            ["path", "description"],
        ),
        (
            "model_runs",
            ["run_id", "model_name", "modality", "cohort_name", "notes"],
            list(model_run_rows(args.include_dev_results) or []),
            ["model_name", "modality", "cohort_name", "notes"],
        ),
        (
            "model_metrics",
            ["run_id", "metric_name", "metric_value", "metric_text"],
            list(metric_rows(args.include_dev_results) or []),
            ["metric_value", "metric_text"],
        ),
        (
            "model_predictions",
            ["run_id", "case_submitter_id", "fold", "risk_score", "os_days", "os_event", "project_id", "risk_group"],
            list(prediction_rows(args.include_dev_results) or []),
            ["fold", "risk_score", "os_days", "os_event", "project_id", "risk_group"],
        ),
        (
            "deg_genes",
            [
                "comparison_name",
                "gene",
                "direction",
                "log2_fold_change",
                "fdr_bh",
                "gbm_mean_tpm",
                "control_mean_tpm",
                "survival_risk_direction",
                "survival_coef",
            ],
            list(deg_rows(ROOT / "results/gbm_deg_proxy.tsv") or [])
            + list(deg_rows(ROOT / "results/gbm_vs_gtex_brain_deg_targeted.tsv") or []),
            [
                "direction",
                "log2_fold_change",
                "fdr_bh",
                "gbm_mean_tpm",
                "control_mean_tpm",
                "survival_risk_direction",
                "survival_coef",
            ],
        ),
        (
            "drug_repurposing_candidates",
            [
                "comparison_name",
                "candidate_rank",
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
                "ranking_note",
            ],
            list(drug_candidate_rows(ROOT / "results/drug_repurposing_candidates.tsv") or [])
            + list(drug_candidate_rows(ROOT / "results/drug_repurposing_candidates_gtex_brain.tsv") or []),
            [
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
                "ranking_note",
            ],
        ),
        (
            "clue_lincs_reversal_candidates",
            [
                "comparison_name",
                "candidate_rank",
                "drug_name",
                "final_repurposing_score",
                "repurposing_score",
                "primary_target",
                "primary_target_direction",
                "local_reversal_prior",
                "local_reversal_reason",
                "clue_tau_score",
                "clue_connectivity_direction",
                "clue_status",
                "approved",
                "antineoplastic",
                "ranking_note",
            ],
            list(clue_candidate_rows(clue_candidates_path) or []),
            [
                "drug_name",
                "final_repurposing_score",
                "repurposing_score",
                "primary_target",
                "primary_target_direction",
                "local_reversal_prior",
                "local_reversal_reason",
                "clue_tau_score",
                "clue_connectivity_direction",
                "clue_status",
                "approved",
                "antineoplastic",
                "ranking_note",
            ],
        ),
        (
            "drug_repurposing_shortlist",
            [
                "comparison_name",
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
            ],
            list(shortlist_rows(ROOT / "results/drug_repurposing_final_shortlist.tsv") or []),
            [
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
            ],
        ),
        (
            "drug_repurposing_validation_matrix",
            [
                "comparison_name",
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
                "pert_id",
                "clue_pert_iname",
                "clue_tau_score",
                "gbm_trial_count",
                "gbm_trial_nct_ids",
                "external_lookup_status",
                "validation_comment_ko",
            ],
            list(validation_matrix_rows(validation_matrix_path) or []),
            [
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
                "pert_id",
                "clue_pert_iname",
                "clue_tau_score",
                "gbm_trial_count",
                "gbm_trial_nct_ids",
                "external_lookup_status",
                "validation_comment_ko",
            ],
        ),
    ]

    with args.out.open("w") as handle:
        handle.write("SET NAMES utf8mb4;\n")
        handle.write("SET FOREIGN_KEY_CHECKS = 0;\n")
        handle.write("START TRANSACTION;\n")
        for table, columns, rows, update_columns in tables:
            for row in rows:
                handle.write(insert_statement(table, columns, row, update_columns))
                handle.write("\n")
        handle.write("COMMIT;\n")
        handle.write("SET FOREIGN_KEY_CHECKS = 1;\n")

    for table, _, rows, _ in tables:
        print(f"{table}: {len(rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
