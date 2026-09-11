#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


def load_cases(path):
    cases = pd.read_csv(path, sep="\t").set_index("case_submitter_id")
    cases["os_days"] = pd.to_numeric(cases["os_days"], errors="coerce")
    cases["os_event"] = pd.to_numeric(cases["os_event"], errors="coerce")
    cases["age_at_diagnosis_days"] = pd.to_numeric(cases["age_at_diagnosis_days"], errors="coerce")
    return cases[cases["os_days"].notna() & cases["os_event"].notna() & cases["os_days"].gt(0)].copy()


def load_clinical_raw(cases, cbio_path):
    clinical = cases[["project_id", "gender", "age_at_diagnosis_days", "tumor_grade", "primary_diagnosis"]].copy()
    clinical["age_at_diagnosis_years"] = clinical["age_at_diagnosis_days"] / 365.25
    clinical = clinical.drop(columns=["age_at_diagnosis_days"])

    if cbio_path and cbio_path.exists():
        cbio = pd.read_csv(cbio_path, sep="\t").set_index("case_submitter_id")
        keep = ["pancan_subtype", "histologic_grade", "cancer_type_detailed", "tumor_type"]
        clinical = clinical.join(cbio[keep], how="left")
        clinical["tumor_grade"] = clinical["tumor_grade"].fillna(clinical["histologic_grade"])
        clinical = clinical.drop(columns=["histologic_grade"])

    categorical = [col for col in clinical.columns if col != "age_at_diagnosis_years"]
    for col in categorical:
        clinical[col] = clinical[col].fillna("Unknown").replace("", "Unknown").astype(str)
    age_median = float(clinical["age_at_diagnosis_years"].median())
    clinical["age_at_diagnosis_years"] = clinical["age_at_diagnosis_years"].fillna(age_median)
    return clinical, categorical, age_median


def fit_clinical_encoder(clinical, categorical):
    return {
        col: sorted(value for value in clinical[col].dropna().astype(str).unique())
        for col in categorical
    }


def transform_clinical(clinical, categorical_levels, age_median):
    clinical = clinical.copy()
    if "age_at_diagnosis_years" not in clinical:
        clinical["age_at_diagnosis_years"] = age_median
    clinical["age_at_diagnosis_years"] = pd.to_numeric(
        clinical["age_at_diagnosis_years"], errors="coerce"
    ).fillna(age_median)

    encoded = pd.DataFrame(index=clinical.index)
    encoded["age_at_diagnosis_years"] = clinical["age_at_diagnosis_years"].astype(float)
    for col, levels in categorical_levels.items():
        values = clinical[col].fillna("Unknown").replace("", "Unknown").astype(str) if col in clinical else pd.Series("Unknown", index=clinical.index)
        for level in levels[1:]:
            encoded[f"{col}_{level}"] = (values == level).astype(float)
    return encoded


def load_rna(path, case_index):
    rna = pd.read_csv(path, sep="\t", index_col=0)
    rna = rna.reindex(case_index).dropna(how="all")
    return np.log2(rna.astype(float) + 1.0)


def select_rna_features(rna, n_features):
    variance = rna.var(axis=0).sort_values(ascending=False)
    return variance.head(min(n_features, len(variance))).index.tolist()


def fit_final_model(cases, clinical, rna, selected_genes, penalizer, l1_ratio):
    x_raw = pd.concat([clinical, rna[selected_genes]], axis=1)
    feature_columns = x_raw.columns[x_raw.var(axis=0) > 0].tolist()
    x_raw = x_raw[feature_columns]

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x_raw), index=x_raw.index, columns=feature_columns)
    train_frame = pd.concat([cases[["os_days", "os_event"]], x], axis=1)
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    model.fit(train_frame, duration_col="os_days", event_col="os_event")
    risk = model.predict_partial_hazard(x).astype(float)
    return model, scaler, feature_columns, risk


def save_example_inputs(out_dir, cases, rna_raw, clinical_raw, case_id):
    example_dir = out_dir / "example_inputs"
    example_dir.mkdir(parents=True, exist_ok=True)
    if case_id not in cases.index:
        case_id = cases.index[0]

    rna_long = (
        rna_raw.loc[case_id]
        .rename_axis("gene")
        .reset_index(name="tpm")
    )
    rna_long.to_csv(example_dir / "example_rna.tsv", sep="\t", index=False)

    clinical_row = clinical_raw.loc[case_id].to_dict()
    clinical_row["case_submitter_id"] = case_id
    clinical_row["age_at_diagnosis_years"] = float(clinical_row["age_at_diagnosis_years"])
    with (example_dir / "example_clinical.json").open("w") as handle:
        json.dump(clinical_row, handle, indent=2)


def maybe_log_mlflow(args, metrics_path, artifact_dir):
    if not args.mlflow:
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping MLflow logging.")
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    metrics = pd.read_csv(metrics_path, sep="\t", header=None, names=["key", "value"])
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "project": "TCGA-LGG/GBM",
                "stage": "final_inference_model",
                "modality": "RNA + Clinical",
            }
        )
        for _, row in metrics.iterrows():
            try:
                value = float(row["value"])
            except ValueError:
                mlflow.log_param(str(row["key"]), str(row["value"]))
            else:
                mlflow.log_metric(str(row["key"]), value)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="final_model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv"))
    parser.add_argument("--rna", type=Path, default=Path("data/processed/rna_tpm_primary_overlap.tsv"))
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/rna_clinical_final"))
    parser.add_argument("--n-rna-features", type=int, default=300)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--example-case", default="")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="TCGA_Glioma_Survival")
    parser.add_argument("--run-name", default="Final RNA+Clinical inference model")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    clinical_raw, categorical, age_median = load_clinical_raw(cases, args.cbio)
    rna_raw = pd.read_csv(args.rna, sep="\t", index_col=0)
    rna = load_rna(args.rna, cases.index)

    common = cases.index.intersection(rna.dropna(how="all").index)
    cases = cases.loc[common].copy()
    clinical_raw = clinical_raw.loc[common].copy()
    rna_raw = rna_raw.loc[common].copy()
    rna = rna.loc[common].fillna(0)

    categorical_levels = fit_clinical_encoder(clinical_raw, categorical)
    clinical = transform_clinical(clinical_raw, categorical_levels, age_median)
    selected_genes = select_rna_features(rna, args.n_rna_features)
    model, scaler, feature_columns, risk = fit_final_model(
        cases,
        clinical,
        rna,
        selected_genes,
        args.penalizer,
        args.l1_ratio,
    )

    predictions = cases[["project_id", "os_days", "os_event"]].copy()
    predictions["risk_score"] = risk
    median_risk = float(predictions["risk_score"].median())
    predictions["risk_group"] = np.where(predictions["risk_score"] >= median_risk, "High", "Low")

    cindex = concordance_index(predictions["os_days"], -predictions["risk_score"], predictions["os_event"])
    high = predictions[predictions["risk_group"].eq("High")]
    low = predictions[predictions["risk_group"].eq("Low")]
    logrank = logrank_test(
        high["os_days"],
        low["os_days"],
        event_observed_A=high["os_event"],
        event_observed_B=low["os_event"],
    )

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "selected_genes": selected_genes,
        "categorical_levels": categorical_levels,
        "age_median": age_median,
        "median_risk": median_risk,
    }
    joblib.dump(bundle, args.out_dir / "model.joblib")
    predictions.to_csv(args.out_dir / "train_predictions.tsv", sep="\t")
    pd.DataFrame({"gene": selected_genes}).to_csv(args.out_dir / "selected_genes.tsv", sep="\t", index=False)
    model.summary.reset_index().rename(columns={"covariate": "feature"}).to_csv(
        args.out_dir / "cox_coefficients.tsv", sep="\t", index=False
    )

    metrics = {
        "n_cases": len(predictions),
        "n_features": len(feature_columns),
        "n_selected_genes": len(selected_genes),
        "penalizer": args.penalizer,
        "l1_ratio": args.l1_ratio,
        "apparent_c_index": float(cindex),
        "median_risk": median_risk,
        "logrank_p": float(logrank.p_value),
    }
    pd.Series(metrics).to_csv(args.out_dir / "metrics.tsv", sep="\t", header=False)

    metadata = {
        "model_type": "CoxPHFitter",
        "modality": "RNA + Clinical",
        "rna_transform": "log2(TPM + 1)",
        "clinical_encoding": "one-hot categorical variables with first sorted level dropped",
        "risk_group_rule": "High if risk_score >= training median_risk",
        "metrics": metrics,
    }
    with (args.out_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    example_case = args.example_case or predictions.sort_values("risk_score", ascending=False).index[0]
    save_example_inputs(args.out_dir, cases, rna_raw, clinical_raw, example_case)
    maybe_log_mlflow(args, args.out_dir / "metrics.tsv", args.out_dir)
    print(
        f"saved final RNA+Clinical model to {args.out_dir} "
        f"(cases={len(predictions)}, features={len(feature_columns)}, apparent_c_index={cindex:.3f})"
    )


if __name__ == "__main__":
    main()
