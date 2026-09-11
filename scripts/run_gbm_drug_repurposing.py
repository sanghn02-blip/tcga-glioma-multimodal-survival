#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats


DGIDB_URL = "https://dgidb.org/api/graphql"
INHIBITORY_TERMS = ("inhibitor", "antagonist", "blocker", "antibody", "suppressor", "negative modulator")
ACTIVATING_TERMS = ("agonist", "activator", "positive modulator", "stimulator")


def bh_fdr(p_values):
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if not finite.any():
        return adjusted
    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    adjusted[finite] = tmp
    return adjusted


def minmax(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low, high = values.min(), values.max()
    if high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def load_expression_groups(rna_path, cases_path, control_mode, normal_matrix):
    rna = pd.read_csv(rna_path, sep="\t")
    cases = pd.read_csv(cases_path, sep="\t")
    rna = rna.merge(cases[["case_submitter_id", "project_id"]], on="case_submitter_id", how="inner")
    genes = [col for col in rna.columns if col not in {"case_submitter_id", "project_id"}]
    gbm = rna[rna["project_id"] == "TCGA-GBM"][genes].apply(pd.to_numeric, errors="coerce")

    if control_mode == "normal_matrix":
        if normal_matrix is None:
            raise ValueError("--normal-matrix is required when --control-mode normal_matrix is used.")
        control = pd.read_csv(normal_matrix, sep="\t")
        if "case_submitter_id" in control.columns:
            control_genes = [col for col in control.columns if col != "case_submitter_id"]
        else:
            control_genes = list(control.columns)
        genes = [gene for gene in genes if gene in control_genes]
        gbm = gbm[genes]
        control = control[genes].apply(pd.to_numeric, errors="coerce")
        comparison_label = "TCGA-GBM vs normal brain"
    else:
        control = rna[rna["project_id"] == "TCGA-LGG"][genes].apply(pd.to_numeric, errors="coerce")
        comparison_label = "TCGA-GBM vs TCGA-LGG proxy"

    return gbm, control, genes, comparison_label


def differential_expression(gbm, control, genes, min_mean_tpm):
    gbm_log = np.log2(gbm[genes].fillna(0).clip(lower=0) + 1.0)
    control_log = np.log2(control[genes].fillna(0).clip(lower=0) + 1.0)
    gbm_mean_raw = gbm[genes].fillna(0).mean(axis=0)
    control_mean_raw = control[genes].fillna(0).mean(axis=0)
    keep = (gbm_mean_raw >= min_mean_tpm) | (control_mean_raw >= min_mean_tpm)
    genes = [gene for gene in genes if bool(keep.get(gene, False))]

    t_stat, p_value = stats.ttest_ind(
        gbm_log[genes],
        control_log[genes],
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    out = pd.DataFrame(
        {
            "gene": genes,
            "gbm_n": int(len(gbm_log)),
            "control_n": int(len(control_log)),
            "gbm_mean_tpm": gbm_mean_raw[genes].values,
            "control_mean_tpm": control_mean_raw[genes].values,
            "gbm_mean_log2_tpm": gbm_log[genes].mean(axis=0).values,
            "control_mean_log2_tpm": control_log[genes].mean(axis=0).values,
            "t_stat": t_stat,
            "p_value": p_value,
        }
    )
    out["log2_fold_change"] = out["gbm_mean_log2_tpm"] - out["control_mean_log2_tpm"]
    out["fdr_bh"] = bh_fdr(out["p_value"])
    out["direction"] = np.where(out["log2_fold_change"] >= 0, "up_in_gbm", "down_in_gbm")
    out["abs_log2_fold_change"] = out["log2_fold_change"].abs()
    out["deg_score"] = out["abs_log2_fold_change"] * -np.log10(out["fdr_bh"].clip(lower=1e-300))
    return out.sort_values(["fdr_bh", "abs_log2_fold_change"], ascending=[True, False])


def load_survival_genes(path):
    if not path.exists():
        return pd.DataFrame(columns=["gene", "survival_coef", "survival_p_value", "survival_risk_direction", "survival_abs_coef"])
    frame = pd.read_csv(path, sep="\t")
    cols = {
        "feature_name": "gene",
        "coef": "survival_coef",
        "p_value": "survival_p_value",
        "risk_direction": "survival_risk_direction",
        "abs_coef": "survival_abs_coef",
    }
    frame = frame.rename(columns=cols)
    keep = [col for col in cols.values() if col in frame.columns]
    return frame[keep].drop_duplicates("gene")


def query_dgidb(genes, timeout, batch_size):
    query = """
    query Genes($names: [String!]!) {
      genes(names: $names) {
        nodes {
          name
          interactions {
            drug { name conceptId approved antiNeoplastic immunotherapy }
            interactionScore
            interactionTypes { type directionality }
            sources { sourceDbName }
            publications { pmid }
          }
        }
      }
    }
    """
    rows = []
    for start in range(0, len(genes), batch_size):
        batch = genes[start : start + batch_size]
        response = requests.post(
            DGIDB_URL,
            json={"query": query, "variables": {"names": batch}},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(json.dumps(payload["errors"]))
        for gene_node in payload.get("data", {}).get("genes", {}).get("nodes", []):
            gene = gene_node.get("name")
            for interaction in gene_node.get("interactions", []) or []:
                drug = interaction.get("drug") or {}
                types = interaction.get("interactionTypes") or []
                sources = interaction.get("sources") or []
                publications = interaction.get("publications") or []
                rows.append(
                    {
                        "gene": gene,
                        "drug_name": drug.get("name"),
                        "drug_concept_id": drug.get("conceptId"),
                        "drug_approved": drug.get("approved"),
                        "drug_antineoplastic": drug.get("antiNeoplastic"),
                        "drug_immunotherapy": drug.get("immunotherapy"),
                        "interaction_score": interaction.get("interactionScore"),
                        "interaction_types": ";".join(sorted({item.get("type", "") for item in types if item.get("type")})),
                        "interaction_directionalities": ";".join(
                            sorted({item.get("directionality", "") for item in types if item.get("directionality")})
                        ),
                        "source_count": len({item.get("sourceDbName") for item in sources if item.get("sourceDbName")}),
                        "sources": ";".join(sorted({item.get("sourceDbName", "") for item in sources if item.get("sourceDbName")})),
                        "pmid_count": len({str(item.get("pmid")) for item in publications if item.get("pmid")}),
                    }
                )
        print(f"[{min(start + batch_size, len(genes))}/{len(genes)}] DGIdb genes queried")
    return pd.DataFrame(rows)


def mechanism_match(row):
    direction = str(row.get("direction", ""))
    types = str(row.get("interaction_types", "")).lower()
    is_inhibitory = any(term in types for term in INHIBITORY_TERMS)
    is_activating = any(term in types for term in ACTIVATING_TERMS)
    if direction == "up_in_gbm" and is_inhibitory:
        return 1.0
    if direction == "down_in_gbm" and is_activating:
        return 1.0
    if is_inhibitory or is_activating:
        return 0.35
    return 0.5


def rank_candidates(deg, interactions, max_rows):
    merged = interactions.merge(deg, on="gene", how="inner")
    if merged.empty:
        return merged
    merged["interaction_score"] = pd.to_numeric(merged["interaction_score"], errors="coerce").fillna(0.0)
    merged["source_count"] = pd.to_numeric(merged["source_count"], errors="coerce").fillna(0.0)
    merged["pmid_count"] = pd.to_numeric(merged["pmid_count"], errors="coerce").fillna(0.0)
    merged["survival_abs_coef"] = pd.to_numeric(merged["survival_abs_coef"], errors="coerce").fillna(0.0)
    merged["drug_approved"] = merged["drug_approved"].fillna(False).astype(bool)
    merged["drug_antineoplastic"] = merged["drug_antineoplastic"].fillna(False).astype(bool)
    merged["drug_immunotherapy"] = merged["drug_immunotherapy"].fillna(False).astype(bool)
    merged["mechanism_match"] = merged.apply(mechanism_match, axis=1)

    merged["deg_score_norm"] = minmax(merged["deg_score"])
    merged["survival_score_norm"] = minmax(merged["survival_abs_coef"])
    merged["interaction_score_norm"] = minmax(merged["interaction_score"])
    merged["evidence_score_norm"] = np.minimum(1.0, (merged["source_count"] + merged["pmid_count"]) / 8.0)
    merged["approval_score"] = (
        0.70 * merged["drug_approved"].astype(float)
        + 0.20 * merged["drug_antineoplastic"].astype(float)
        + 0.10 * merged["drug_immunotherapy"].astype(float)
    ).clip(upper=1.0)
    merged["gene_drug_score"] = (
        0.31 * merged["deg_score_norm"]
        + 0.18 * merged["survival_score_norm"]
        + 0.16 * merged["interaction_score_norm"]
        + 0.13 * merged["evidence_score_norm"]
        + 0.10 * merged["mechanism_match"]
        + 0.12 * merged["approval_score"]
    )

    grouped = []
    for drug, group in merged.groupby("drug_name"):
        group = group.sort_values("gene_drug_score", ascending=False)
        top = group.iloc[0]
        score = 0.72 * group["gene_drug_score"].max() + 0.28 * group["gene_drug_score"].head(3).mean()
        grouped.append(
            {
                "rank": 0,
                "drug_name": drug,
                "repurposing_score": float(score),
                "primary_target": top["gene"],
                "targets": ";".join(group["gene"].drop_duplicates().head(8)),
                "target_count": int(group["gene"].nunique()),
                "primary_target_direction": top["direction"],
                "primary_target_log2_fc": top["log2_fold_change"],
                "primary_target_fdr": top["fdr_bh"],
                "primary_target_survival_direction": top.get("survival_risk_direction", ""),
                "primary_target_survival_coef": top.get("survival_coef", ""),
                "interaction_types": top.get("interaction_types", ""),
                "sources": ";".join(sorted(set(";".join(group["sources"].fillna("")).split(";")) - {""})),
                "source_count_total": int(group["source_count"].sum()),
                "pmid_count_total": int(group["pmid_count"].sum()),
                "approved": bool(group["drug_approved"].max()),
                "antineoplastic": bool(group["drug_antineoplastic"].max()),
                "immunotherapy": bool(group["drug_immunotherapy"].max()),
                "mechanism_match": float(group["mechanism_match"].max()),
            }
        )
    out = pd.DataFrame(grouped).sort_values("repurposing_score", ascending=False).head(max_rows).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, default=Path("data/processed/rna_tpm_primary_overlap.tsv"))
    parser.add_argument("--cases", type=Path, default=Path("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv"))
    parser.add_argument("--control-mode", choices=["lgg_proxy", "normal_matrix"], default="lgg_proxy")
    parser.add_argument("--normal-matrix", type=Path)
    parser.add_argument("--survival-genes", type=Path, default=Path("results/interpretability_gbm_rna_clinical_fast/rna_gene_coefficients.tsv"))
    parser.add_argument("--out-deg", type=Path, default=Path("results/gbm_deg_proxy.tsv"))
    parser.add_argument("--out-interactions", type=Path, default=Path("results/dgidb_gbm_target_interactions.tsv"))
    parser.add_argument("--out-candidates", type=Path, default=Path("results/drug_repurposing_candidates.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/drug_repurposing_summary.json"))
    parser.add_argument("--top-genes", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--min-mean-tpm", type=float, default=1.0)
    parser.add_argument("--min-abs-log2-fc", type=float, default=0.8)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--dgidb-timeout", type=int, default=60)
    parser.add_argument("--dgidb-batch-size", type=int, default=25)
    parser.add_argument("--skip-dgidb", action="store_true")
    parser.add_argument("--include-unapproved", action="store_true")
    args = parser.parse_args()

    gbm, control, genes, comparison_label = load_expression_groups(args.rna, args.cases, args.control_mode, args.normal_matrix)
    deg = differential_expression(gbm, control, genes, args.min_mean_tpm)
    survival = load_survival_genes(args.survival_genes)
    deg_with_survival = deg.merge(survival, on="gene", how="left")
    deg_with_survival["survival_abs_coef"] = pd.to_numeric(deg_with_survival["survival_abs_coef"], errors="coerce").fillna(0.0)
    deg_with_survival["comparison"] = comparison_label
    args.out_deg.parent.mkdir(parents=True, exist_ok=True)
    deg_with_survival.to_csv(args.out_deg, sep="\t", index=False)

    prioritized = deg_with_survival[
        (deg_with_survival["fdr_bh"] <= args.max_fdr)
        & (deg_with_survival["abs_log2_fold_change"] >= args.min_abs_log2_fc)
    ].copy()
    prioritized["priority"] = (
        prioritized["deg_score"].rank(pct=True)
        + prioritized["survival_abs_coef"].rank(pct=True)
        + (prioritized["direction"] == "up_in_gbm").astype(float) * 0.15
    )
    gene_list = prioritized.sort_values("priority", ascending=False)["gene"].head(args.top_genes).tolist()

    interactions = pd.DataFrame()
    if gene_list and not args.skip_dgidb:
        interactions = query_dgidb(gene_list, args.dgidb_timeout, args.dgidb_batch_size)
    interactions.to_csv(args.out_interactions, sep="\t", index=False)

    rankable_interactions = interactions
    if not args.include_unapproved and not interactions.empty and "drug_approved" in interactions.columns:
        rankable_interactions = interactions[interactions["drug_approved"].fillna(False).astype(bool)].copy()
    candidates = rank_candidates(deg_with_survival, rankable_interactions, args.max_candidates)
    candidates["comparison"] = comparison_label if not candidates.empty else comparison_label
    candidates["ranking_note"] = (
        "Target-based drug repurposing prototype. Use GTEx normal brain comparison before clinical interpretation."
        if args.control_mode == "lgg_proxy"
        else "Target-based drug repurposing ranking from GBM-vs-normal expression and DGIdb."
    )
    candidates.to_csv(args.out_candidates, sep="\t", index=False)

    summary = {
        "comparison": comparison_label,
        "gbm_samples": int(len(gbm)),
        "control_samples": int(len(control)),
        "genes_tested": int(len(deg)),
        "prioritized_genes": int(len(gene_list)),
        "dgidb_interactions": int(len(interactions)),
        "rankable_interactions": int(len(rankable_interactions)),
        "approved_only": not args.include_unapproved,
        "drug_candidates": int(len(candidates)),
        "out_deg": str(args.out_deg),
        "out_interactions": str(args.out_interactions),
        "out_candidates": str(args.out_candidates),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
