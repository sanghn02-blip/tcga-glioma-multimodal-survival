#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    {
        "cohort": "All LGG+GBM",
        "analysis": "RNA-only interpretation",
        "cases": "outputs/tcga_lgg_gbm_primary_overlap_cases.tsv",
        "selected": "results/full_rna_baseline_fast/selected_features.tsv",
        "predictions": "results/full_rna_baseline_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_all_rna_fast",
        "adjust": False,
    },
    {
        "cohort": "All LGG+GBM",
        "analysis": "RNA adjusted for clinical",
        "cases": "outputs/tcga_lgg_gbm_primary_overlap_cases.tsv",
        "selected": "results/full_rna_clinical_baseline_fast/selected_features.tsv",
        "predictions": "results/full_rna_clinical_baseline_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_all_rna_clinical_fast",
        "adjust": True,
    },
    {
        "cohort": "LGG-only",
        "analysis": "RNA-only interpretation",
        "cases": "outputs/stratified_case_sets/tcga_lgg.tsv",
        "selected": "results/stratified_tcga_lgg_rna_fast/selected_features.tsv",
        "predictions": "results/stratified_tcga_lgg_rna_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_lgg_rna_fast",
        "adjust": False,
    },
    {
        "cohort": "LGG-only",
        "analysis": "RNA adjusted for clinical",
        "cases": "outputs/stratified_case_sets/tcga_lgg.tsv",
        "selected": "results/stratified_tcga_lgg_rna_clinical_fast/selected_features.tsv",
        "predictions": "results/stratified_tcga_lgg_rna_clinical_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_lgg_rna_clinical_fast",
        "adjust": True,
    },
    {
        "cohort": "GBM-only",
        "analysis": "RNA-only interpretation",
        "cases": "outputs/stratified_case_sets/tcga_gbm.tsv",
        "selected": "results/stratified_tcga_gbm_rna_fast/selected_features.tsv",
        "predictions": "results/stratified_tcga_gbm_rna_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_gbm_rna_fast",
        "adjust": False,
    },
    {
        "cohort": "GBM-only",
        "analysis": "RNA adjusted for clinical",
        "cases": "outputs/stratified_case_sets/tcga_gbm.tsv",
        "selected": "results/stratified_tcga_gbm_rna_clinical_fast/selected_features.tsv",
        "predictions": "results/stratified_tcga_gbm_rna_clinical_fast/cox_predictions.tsv",
        "out_dir": "results/interpretability_gbm_rna_clinical_fast",
        "adjust": True,
    },
]


def run_interpretability(config, args):
    command = [
        sys.executable,
        "scripts/run_rna_interpretability.py",
        "--cases",
        config["cases"],
        "--rna",
        args.rna,
        "--selected-features",
        config["selected"],
        "--predictions",
        config["predictions"],
        "--out-dir",
        config["out_dir"],
        "--top-n",
        str(args.top_n),
        "--patient-top-k",
        str(args.patient_top_k),
        "--penalizer",
        str(args.penalizer),
    ]
    if config["adjust"]:
        command.append("--adjust-clinical")
    subprocess.run(command, cwd=ROOT, check=True)


def write_top_gene_summary(out):
    rows = []
    for config in CONFIGS:
        path = ROOT / config["out_dir"] / "rna_gene_coefficients.tsv"
        df = pd.read_csv(path, sep="\t")
        keep = df.sort_values(["selection_count", "abs_coef"], ascending=[False, False]).head(15)
        for _, row in keep.iterrows():
            rows.append(
                {
                    "cohort": config["cohort"],
                    "analysis": config["analysis"],
                    "gene": row["feature_name"],
                    "coef": row["coef"],
                    "hazard_ratio": row["hazard_ratio"],
                    "p_value": row["p_value"],
                    "selection_count": int(row["selection_count"]),
                    "selection_rate": row["selection_rate"],
                    "risk_direction": row["risk_direction"],
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / out, sep="\t", index=False)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", default="data/processed/rna_tpm_primary_overlap.tsv")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--patient-top-k", type=int, default=5)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("results/rna_interpretability_top_genes.tsv"))
    args = parser.parse_args()

    for config in CONFIGS:
        print(f"running {config['cohort']} - {config['analysis']}", flush=True)
        run_interpretability(config, args)
    write_top_gene_summary(args.out)


if __name__ == "__main__":
    main()
