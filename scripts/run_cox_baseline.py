#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_cases(path):
    cases = pd.read_csv(path, sep="\t").set_index("case_submitter_id")
    cases["os_days"] = pd.to_numeric(cases["os_days"], errors="coerce")
    cases["os_event"] = pd.to_numeric(cases["os_event"], errors="coerce")
    cases["age_at_diagnosis_days"] = pd.to_numeric(cases["age_at_diagnosis_days"], errors="coerce")
    cases = cases[cases["os_days"].notna() & cases["os_event"].notna() & cases["os_days"].gt(0)].copy()
    return cases


def load_clinical_features(cases, cbio_path=None):
    clinical = cases[["project_id", "gender", "age_at_diagnosis_days", "tumor_grade", "primary_diagnosis"]].copy()
    clinical["age_at_diagnosis_years"] = clinical["age_at_diagnosis_days"] / 365.25
    clinical = clinical.drop(columns=["age_at_diagnosis_days"])

    if cbio_path:
        cbio = pd.read_csv(cbio_path, sep="\t").set_index("case_submitter_id")
        keep = ["pancan_subtype", "histologic_grade", "cancer_type_detailed", "tumor_type"]
        clinical = clinical.join(cbio[keep], how="left")
        clinical["tumor_grade"] = clinical["tumor_grade"].fillna(clinical["histologic_grade"])
        clinical = clinical.drop(columns=["histologic_grade"])

    categorical = [col for col in clinical.columns if col != "age_at_diagnosis_years"]
    for col in categorical:
        clinical[col] = clinical[col].fillna("Unknown").replace("", "Unknown")
    clinical["age_at_diagnosis_years"] = clinical["age_at_diagnosis_years"].fillna(clinical["age_at_diagnosis_years"].median())
    encoded = pd.get_dummies(clinical, columns=categorical, drop_first=True, dtype=float)
    return encoded


def load_rna_features(path, case_index):
    if not path:
        return pd.DataFrame(index=case_index)
    rna = pd.read_csv(path, sep="\t", index_col=0)
    rna = rna.reindex(case_index).dropna(how="all")
    rna = np.log2(rna.astype(float) + 1.0)
    return rna


def load_wsi_features(path, case_index):
    if not path:
        return pd.DataFrame(index=case_index)
    wsi = pd.read_csv(path, sep="\t").set_index("case_submitter_id")
    feature_cols = [col for col in wsi.columns if col.startswith(("wsi_", "emb_"))]
    wsi = wsi[feature_cols].apply(pd.to_numeric, errors="coerce")
    wsi = wsi.groupby(level=0).mean()
    return wsi.reindex(case_index).dropna(how="all")


def select_rna_features(train_rna, n_features):
    if train_rna.empty:
        return []
    variance = train_rna.var(axis=0).sort_values(ascending=False)
    return variance.head(min(n_features, len(variance))).index.tolist()


def select_wsi_features(train_wsi, n_features):
    if train_wsi.empty:
        return []
    if not n_features or n_features >= train_wsi.shape[1]:
        return train_wsi.columns.tolist()
    variance = train_wsi.var(axis=0).sort_values(ascending=False)
    return variance.head(min(n_features, len(variance))).index.tolist()


def prepare_fold_features(train_base, test_base):
    nonzero = train_base.columns[train_base.var(axis=0) > 0].tolist()
    train_base = train_base[nonzero]
    test_base = test_base.reindex(columns=nonzero, fill_value=0)
    scaler = StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_base), index=train_base.index, columns=nonzero)
    test_x = pd.DataFrame(scaler.transform(test_base), index=test_base.index, columns=nonzero)
    return train_x, test_x


def fit_cox(frame, duration_col, event_col, penalizer, l1_ratio):
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    model.fit(frame, duration_col=duration_col, event_col=event_col)
    return model


def build_km(pred):
    km = KaplanMeierFitter()
    rows = []
    for group, group_frame in pred.groupby("risk_group"):
        km.fit(group_frame["os_days"], event_observed=group_frame["os_event"], label=group)
        survival = km.survival_function_.reset_index()
        survival.columns = ["timeline", "survival_probability"]
        survival["risk_group"] = group
        rows.append(survival)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--rna", type=Path)
    parser.add_argument("--wsi", type=Path)
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument(
        "--feature-set",
        choices=["clinical", "rna", "rna_clinical", "wsi", "wsi_clinical", "wsi_rna", "wsi_rna_clinical"],
        required=True,
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-rna-features", type=int, default=1000)
    parser.add_argument("--n-wsi-features", type=int, default=0, help="0 keeps all WSI/embedding features.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    uses_clinical = args.feature_set in {"clinical", "rna_clinical", "wsi_clinical", "wsi_rna_clinical"}
    uses_rna = args.feature_set in {"rna", "rna_clinical", "wsi_rna", "wsi_rna_clinical"}
    uses_wsi = args.feature_set in {"wsi", "wsi_clinical", "wsi_rna", "wsi_rna_clinical"}
    clinical = load_clinical_features(cases, args.cbio if uses_clinical else None)
    rna = load_rna_features(args.rna, cases.index) if uses_rna else pd.DataFrame(index=cases.index)
    wsi = load_wsi_features(args.wsi, cases.index) if uses_wsi else pd.DataFrame(index=cases.index)

    common = cases.index
    if uses_rna:
        common = common.intersection(rna.dropna(how="all").index)
    if uses_wsi:
        common = common.intersection(wsi.dropna(how="all").index)
    cases = cases.loc[common].copy()
    clinical = clinical.reindex(common).fillna(0)
    rna = rna.reindex(common).fillna(0)
    wsi = wsi.reindex(common).fillna(0)

    fold_count = min(args.folds, len(cases))
    if fold_count < 2:
        raise ValueError("Need at least two cases for cross-validation.")

    predictions = []
    selected_rows = []
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(cases), 1):
        train_cases = cases.iloc[train_idx]
        test_cases = cases.iloc[test_idx]
        train_parts = []
        test_parts = []

        if uses_clinical:
            train_parts.append(clinical.iloc[train_idx])
            test_parts.append(clinical.iloc[test_idx])
            selected_rows.extend(
                {"fold": fold, "feature_name": col, "feature_source": "clinical"}
                for col in clinical.columns
            )

        if uses_rna:
            selected_rna = select_rna_features(rna.iloc[train_idx], args.n_rna_features)
            train_parts.append(rna.iloc[train_idx][selected_rna])
            test_parts.append(rna.iloc[test_idx][selected_rna])
            selected_rows.extend(
                {"fold": fold, "feature_name": col, "feature_source": "rna"}
                for col in selected_rna
            )

        if uses_wsi:
            selected_wsi = select_wsi_features(wsi.iloc[train_idx], args.n_wsi_features)
            train_parts.append(wsi.iloc[train_idx][selected_wsi])
            test_parts.append(wsi.iloc[test_idx][selected_wsi])
            selected_rows.extend(
                {"fold": fold, "feature_name": col, "feature_source": "wsi"}
                for col in selected_wsi
            )

        train_base = pd.concat(train_parts, axis=1)
        test_base = pd.concat(test_parts, axis=1)
        train_x, test_x = prepare_fold_features(train_base, test_base)
        train_frame = pd.concat([train_cases[["os_days", "os_event"]], train_x], axis=1)
        model = fit_cox(train_frame, "os_days", "os_event", args.penalizer, args.l1_ratio)
        risk = model.predict_partial_hazard(test_x).astype(float)

        for case_id, score in risk.items():
            predictions.append(
                {
                    "case_submitter_id": case_id,
                    "fold": fold,
                    "risk_score": score,
                    "os_days": test_cases.loc[case_id, "os_days"],
                    "os_event": test_cases.loc[case_id, "os_event"],
                    "project_id": test_cases.loc[case_id, "project_id"],
                }
            )

    pred = pd.DataFrame(predictions).set_index("case_submitter_id")
    median_risk = pred["risk_score"].median()
    pred["risk_group"] = np.where(pred["risk_score"] >= median_risk, "High", "Low")
    cindex = concordance_index(pred["os_days"], -pred["risk_score"], pred["os_event"])

    high = pred[pred["risk_group"].eq("High")]
    low = pred[pred["risk_group"].eq("Low")]
    logrank = logrank_test(
        high["os_days"],
        low["os_days"],
        event_observed_A=high["os_event"],
        event_observed_B=low["os_event"],
    )

    pred.to_csv(args.out_dir / "cox_predictions.tsv", sep="\t")
    build_km(pred).to_csv(args.out_dir / "cox_km_curve.tsv", sep="\t", index=False)
    pd.DataFrame(selected_rows).drop_duplicates().to_csv(args.out_dir / "selected_features.tsv", sep="\t", index=False)

    metrics = {
        "feature_set": args.feature_set,
        "n_cases": len(pred),
        "n_clinical_features": clinical.shape[1] if uses_clinical else 0,
        "n_rna_features_requested": args.n_rna_features if uses_rna else 0,
        "n_wsi_features_requested": args.n_wsi_features if uses_wsi else 0,
        "n_wsi_features": wsi.shape[1] if uses_wsi else 0,
        "folds": fold_count,
        "penalizer": args.penalizer,
        "l1_ratio": args.l1_ratio,
        "c_index_cv": cindex,
        "median_risk": median_risk,
        "logrank_p": logrank.p_value,
    }
    pd.Series(metrics).to_csv(args.out_dir / "cox_metrics.tsv", sep="\t", header=False)
    print(f"feature_set={args.feature_set} cases={len(pred)} c_index_cv={cindex:.3f} logrank_p={logrank.p_value:.4g}")


if __name__ == "__main__":
    main()
