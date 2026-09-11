#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def cindex_from_scores(time, event, risk):
    from lifelines.utils import concordance_index

    return concordance_index(time, -risk, event)


def select_features(x, y, n_features):
    variance = x.var(axis=0).sort_values(ascending=False)
    top = variance.head(min(n_features, len(variance))).index.tolist()
    return top


def fit_cox(train, duration_col, event_col, penalizer):
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5)
    model.fit(train, duration_col=duration_col, event_col=event_col)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--penalizer", type=float, default=0.1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rna = pd.read_csv(args.rna, sep="\t", index_col=0)
    cases = pd.read_csv(args.cases, sep="\t").set_index("case_submitter_id")
    joined = cases.join(rna, how="inner")
    joined = joined[joined["os_days"].notna() & joined["os_event"].notna()].copy()
    joined["os_days"] = pd.to_numeric(joined["os_days"])
    joined["os_event"] = pd.to_numeric(joined["os_event"])
    joined = joined[joined["os_days"] > 0]

    gene_cols = [col for col in rna.columns if col in joined.columns]
    x = np.log2(joined[gene_cols].astype(float) + 1.0)
    y = joined[["os_days", "os_event"]]
    selected = select_features(x, y, args.n_features)

    fold_count = min(args.folds, len(joined))
    if fold_count < 2:
        raise ValueError("Need at least two cases for cross-validation.")

    predictions = []
    for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=fold_count, shuffle=True, random_state=42).split(joined), 1):
        train_cases = joined.iloc[train_idx].copy()
        test_cases = joined.iloc[test_idx].copy()
        scaler = StandardScaler()
        train_x = pd.DataFrame(
            scaler.fit_transform(x.iloc[train_idx][selected]),
            index=train_cases.index,
            columns=selected,
        )
        test_x = pd.DataFrame(
            scaler.transform(x.iloc[test_idx][selected]),
            index=test_cases.index,
            columns=selected,
        )
        train_frame = pd.concat([train_cases[["os_days", "os_event"]], train_x], axis=1)
        model = fit_cox(train_frame, "os_days", "os_event", args.penalizer)
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
    cindex = cindex_from_scores(pred["os_days"], pred["os_event"], pred["risk_score"])

    high = pred[pred["risk_group"].eq("High")]
    low = pred[pred["risk_group"].eq("Low")]
    logrank = logrank_test(
        high["os_days"],
        low["os_days"],
        event_observed_A=high["os_event"],
        event_observed_B=low["os_event"],
    )

    pred.to_csv(args.out_dir / "rna_cox_predictions.tsv", sep="\t")
    pd.Series(
        {
            "n_cases": len(pred),
            "n_genes_input": len(gene_cols),
            "n_genes_selected": len(selected),
            "folds": fold_count,
            "c_index_cv": cindex,
            "median_risk": median_risk,
            "logrank_p": logrank.p_value,
        }
    ).to_csv(args.out_dir / "rna_cox_metrics.tsv", sep="\t", header=False)

    km = KaplanMeierFitter()
    km_rows = []
    for group, group_frame in pred.groupby("risk_group"):
        km.fit(group_frame["os_days"], event_observed=group_frame["os_event"], label=group)
        survival = km.survival_function_.reset_index()
        survival.columns = ["timeline", "survival_probability"]
        survival["risk_group"] = group
        km_rows.append(survival)
    pd.concat(km_rows, ignore_index=True).to_csv(args.out_dir / "rna_cox_km_curve.tsv", sep="\t", index=False)

    print(f"cases={len(pred)} c_index_cv={cindex:.3f} logrank_p={logrank.p_value:.4g}")


if __name__ == "__main__":
    main()
