#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_final_survival_model import transform_clinical


def load_rna_input(path, selected_genes):
    frame = pd.read_csv(path, sep="\t")
    lower_columns = {col.lower(): col for col in frame.columns}

    if {"gene", "tpm"}.issubset(lower_columns):
        gene_col = lower_columns["gene"]
        value_col = lower_columns["tpm"]
        series = frame.set_index(gene_col)[value_col]
    elif {"gene_symbol", "tpm"}.issubset(lower_columns):
        gene_col = lower_columns["gene_symbol"]
        value_col = lower_columns["tpm"]
        series = frame.set_index(gene_col)[value_col]
    elif {"gene", "expression"}.issubset(lower_columns):
        gene_col = lower_columns["gene"]
        value_col = lower_columns["expression"]
        series = frame.set_index(gene_col)[value_col]
    else:
        wide = frame.copy()
        if "case_submitter_id" in wide.columns:
            wide = wide.drop(columns=["case_submitter_id"])
        series = wide.iloc[0]

    values = pd.to_numeric(series.reindex(selected_genes), errors="coerce").fillna(0.0)
    return pd.DataFrame([np.log2(values.astype(float) + 1.0)], columns=selected_genes)


def load_clinical_input(path):
    with Path(path).open() as handle:
        record = json.load(handle)
    case_id = record.pop("case_submitter_id", "uploaded_patient")
    if "age_at_diagnosis_years" not in record and "age_at_diagnosis_days" in record:
        record["age_at_diagnosis_years"] = float(record["age_at_diagnosis_days"]) / 365.25
    clinical = pd.DataFrame([record], index=[case_id])
    return case_id, clinical


def top_contributions(x_scaled, model, top_k):
    coef = model.params_.reindex(x_scaled.columns).fillna(0.0)
    contrib = x_scaled.iloc[0].multiply(coef)
    rows = []
    for feature, value in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_k).items():
        rows.append(
            {
                "feature": feature,
                "contribution": float(value),
                "direction": "raises_risk" if value > 0 else "lowers_risk",
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/rna_clinical_final"))
    parser.add_argument("--rna-file", type=Path, required=True)
    parser.add_argument("--clinical-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/predictions/example_patient"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model_dir / "model.joblib")
    case_id, clinical_raw = load_clinical_input(args.clinical_json)
    clinical = transform_clinical(
        clinical_raw,
        bundle["categorical_levels"],
        bundle["age_median"],
    )
    rna = load_rna_input(args.rna_file, bundle["selected_genes"])
    rna.index = clinical.index

    x_raw = pd.concat([clinical, rna], axis=1)
    x_raw = x_raw.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    x_scaled = pd.DataFrame(
        bundle["scaler"].transform(x_raw),
        index=x_raw.index,
        columns=bundle["feature_columns"],
    )
    risk_score = float(bundle["model"].predict_partial_hazard(x_scaled).iloc[0])
    risk_group = "High" if risk_score >= bundle["median_risk"] else "Low"

    available_genes = [gene for gene in bundle["selected_genes"] if gene in x_scaled.columns]
    clinical_features = [feature for feature in x_scaled.columns if feature not in set(available_genes)]
    gene_contrib = top_contributions(x_scaled[available_genes], bundle["model"], args.top_k)
    clinical_contrib = top_contributions(x_scaled[clinical_features], bundle["model"], args.top_k)

    result = {
        "case_submitter_id": case_id,
        "risk_score": risk_score,
        "risk_group": risk_group,
        "training_median_risk": float(bundle["median_risk"]),
        "model_dir": str(args.model_dir),
        "note": "Research prototype output. Not for clinical decision-making.",
    }

    with (args.out_dir / "prediction.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    gene_contrib.to_csv(args.out_dir / "top_gene_contributions.tsv", sep="\t", index=False)
    clinical_contrib.to_csv(args.out_dir / "top_clinical_contributions.tsv", sep="\t", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
