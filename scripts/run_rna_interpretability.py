#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from run_cox_baseline import fit_cox, load_cases, load_clinical_features, load_rna_features, prepare_fold_features


def load_selection_counts(path):
    selected = pd.read_csv(path, sep="\t")
    selected = selected[selected["feature_source"].eq("rna")].copy()
    if selected.empty:
        raise ValueError(f"No RNA features found in {path}")
    n_folds = selected["fold"].nunique()
    counts = (
        selected.groupby("feature_name")
        .agg(selection_count=("fold", "nunique"))
        .reset_index()
        .rename(columns={"feature_name": "gene"})
    )
    counts["selection_rate"] = counts["selection_count"] / n_folds
    return counts.sort_values(["selection_count", "gene"], ascending=[False, True]), n_folds


def summarize_group_contributions(contributions, predictions):
    if not predictions:
        return pd.DataFrame()
    pred = pd.read_csv(predictions, sep="\t").set_index("case_submitter_id")
    common = contributions.index.intersection(pred.index)
    if common.empty or "risk_group" not in pred.columns:
        return pd.DataFrame()

    joined = contributions.loc[common].join(pred[["risk_group"]], how="inner")
    rows = []
    for gene in contributions.columns:
        high = joined.loc[joined["risk_group"].eq("High"), gene]
        low = joined.loc[joined["risk_group"].eq("Low"), gene]
        rows.append(
            {
                "gene": gene,
                "mean_contribution_high_risk_group": high.mean(),
                "mean_contribution_low_risk_group": low.mean(),
                "high_minus_low_mean_contribution": high.mean() - low.mean(),
            }
        )
    return pd.DataFrame(rows)


def top_patient_contributions(contributions, top_k):
    rows = []
    for case_id, values in contributions.iterrows():
        top = values.reindex(values.abs().sort_values(ascending=False).head(top_k).index)
        for rank, (gene, contribution) in enumerate(top.items(), 1):
            rows.append(
                {
                    "case_submitter_id": case_id,
                    "rank": rank,
                    "gene": gene,
                    "risk_contribution": contribution,
                    "direction": "raises_risk" if contribution > 0 else "lowers_risk",
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--rna", required=True, type=Path)
    parser.add_argument("--selected-features", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--patient-top-k", type=int, default=5)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--adjust-clinical", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.cases)
    rna = load_rna_features(args.rna, cases.index)
    common = cases.index.intersection(rna.dropna(how="all").index)
    cases = cases.loc[common].copy()
    rna = rna.reindex(common).fillna(0)

    selection_counts, n_folds = load_selection_counts(args.selected_features)
    selected_genes = selection_counts.head(args.top_n)["gene"].tolist()
    selected_genes = [gene for gene in selected_genes if gene in rna.columns]
    if not selected_genes:
        raise ValueError("No selected RNA genes were present in the RNA matrix.")

    parts = [rna[selected_genes]]
    feature_sources = {gene: "rna" for gene in selected_genes}
    if args.adjust_clinical:
        clinical = load_clinical_features(cases, args.cbio)
        clinical = clinical.reindex(common).fillna(0)
        parts.insert(0, clinical)
        feature_sources.update({col: "clinical" for col in clinical.columns})

    base = pd.concat(parts, axis=1)
    x, _ = prepare_fold_features(base, base)
    frame = pd.concat([cases[["os_days", "os_event"]], x], axis=1)
    model = fit_cox(frame, "os_days", "os_event", args.penalizer, args.l1_ratio)
    risk = model.predict_partial_hazard(x).astype(float)
    cindex = concordance_index(cases["os_days"], -risk, cases["os_event"])

    coef = model.summary.reset_index().rename(
        columns={
            "covariate": "feature_name",
            "exp(coef)": "hazard_ratio",
            "p": "p_value",
        }
    )
    coef["feature_source"] = coef["feature_name"].map(feature_sources).fillna("unknown")
    gene_coef = coef[coef["feature_source"].eq("rna")].copy()
    gene_coef = gene_coef.merge(selection_counts, left_on="feature_name", right_on="gene", how="left")
    gene_coef = gene_coef.drop(columns=["gene"])
    gene_coef["selection_count"] = gene_coef["selection_count"].fillna(0).astype(int)
    gene_coef["selection_rate"] = gene_coef["selection_rate"].fillna(0)
    gene_coef["risk_direction"] = np.where(
        gene_coef["coef"].gt(0),
        "higher_expression_higher_risk",
        "higher_expression_lower_risk",
    )
    gene_coef["abs_coef"] = gene_coef["coef"].abs()
    gene_coef = gene_coef.sort_values(["selection_count", "abs_coef"], ascending=[False, False])

    gene_coef.to_csv(args.out_dir / "rna_gene_coefficients.tsv", sep="\t", index=False)
    selection_counts.to_csv(args.out_dir / "rna_selection_frequency.tsv", sep="\t", index=False)

    rna_columns = gene_coef["feature_name"].tolist()
    contributions = x[rna_columns].multiply(model.params_.reindex(rna_columns), axis=1)
    contributions.to_csv(args.out_dir / "patient_gene_contribution_matrix.tsv", sep="\t")
    top_patient_contributions(contributions, args.patient_top_k).to_csv(
        args.out_dir / "patient_top_gene_contributions.tsv",
        sep="\t",
        index=False,
    )

    group_summary = summarize_group_contributions(contributions, args.predictions)
    if not group_summary.empty:
        group_summary = group_summary.merge(gene_coef[["feature_name", "coef"]], left_on="gene", right_on="feature_name", how="left")
        group_summary = group_summary.drop(columns=["feature_name"])
        group_summary = group_summary.sort_values("high_minus_low_mean_contribution", key=lambda s: s.abs(), ascending=False)
        group_summary.to_csv(args.out_dir / "risk_group_gene_contributions.tsv", sep="\t", index=False)

    metrics = {
        "n_cases": len(cases),
        "n_events": int(cases["os_event"].sum()),
        "n_folds_from_feature_selection": n_folds,
        "n_selected_rna_genes": len(selected_genes),
        "adjust_clinical": args.adjust_clinical,
        "penalizer": args.penalizer,
        "l1_ratio": args.l1_ratio,
        "apparent_c_index": cindex,
    }
    pd.Series(metrics).to_csv(args.out_dir / "interpretability_metrics.tsv", sep="\t", header=False)
    print(
        f"wrote RNA interpretability outputs to {args.out_dir} "
        f"(cases={len(cases)}, genes={len(selected_genes)}, apparent_c_index={cindex:.3f})"
    )


if __name__ == "__main__":
    main()
