#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_gbm_drug_repurposing import bh_fdr, load_survival_genes, query_dgidb, rank_candidates


XENA_HUB = "https://toil.xenahubs.net"
XENA_DATASET = "TcgaTargetGtex_rsem_gene_tpm"
PHENOTYPE_URL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz"
PROBEMAP_URL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fgencode.v23.annotation.gene.probemap"

DATASET_PROBE_VALUES = """(fn [dataset samples probes]
  (let [probemap (:probemap (car (query {:select [:probemap]
                                         :from [:dataset]
                                         :where [:= :name dataset]})))
        position (if probemap
                    ((xena-query {:select ["name" "position"]
                                  :from [probemap]
                                  :where [:in "name" probes]}) "position")
                    nil)]
    [position
     (fetch [{:table dataset
              :columns probes
              :samples samples}])]))"""


def ensure_download(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, path)


def marshall_param(value):
    if value is None:
        return "nil"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + " ".join(marshall_param(item) for item in value) + "]"
    return str(value)


def xena_call(function_body, *params, timeout=120):
    query = "(" + function_body + " " + " ".join(marshall_param(param) for param in params) + ")"
    request = urllib.request.Request(
        XENA_HUB + "/data/",
        data=query.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def select_samples(phenotype, max_gbm, max_normal):
    gbm = phenotype[
        phenotype["_study"].eq("TCGA")
        & phenotype["primary disease or tissue"].eq("Glioblastoma Multiforme")
        & phenotype["_sample_type"].eq("Primary Tumor")
    ].copy()
    normal = phenotype[
        phenotype["_study"].eq("GTEX")
        & phenotype["primary disease or tissue"].astype(str).str.startswith("Brain", na=False)
        & phenotype["_sample_type"].eq("Normal Tissue")
    ].copy()
    if max_gbm:
        gbm = gbm.head(max_gbm)
    if max_normal:
        normal = normal.head(max_normal)
    sample_rows = pd.concat(
        [
            gbm.assign(comparison_group="GBM primary tumor"),
            normal.assign(comparison_group="GTEx normal brain"),
        ],
        ignore_index=True,
    )
    return gbm["sample"].tolist(), normal["sample"].tolist(), sample_rows


def prioritized_genes(proxy_deg, survival_genes, top_genes, min_abs_log2_fc, max_fdr):
    deg = pd.read_csv(proxy_deg, sep="\t")
    survival = load_survival_genes(survival_genes)
    if "survival_abs_coef" not in deg.columns:
        deg = deg.merge(survival, on="gene", how="left")
    deg["survival_abs_coef"] = pd.to_numeric(deg.get("survival_abs_coef"), errors="coerce").fillna(0.0)
    deg = deg[
        (pd.to_numeric(deg["fdr_bh"], errors="coerce") <= max_fdr)
        & (pd.to_numeric(deg["abs_log2_fold_change"], errors="coerce") >= min_abs_log2_fc)
    ].copy()
    deg["priority"] = (
        deg["deg_score"].rank(pct=True)
        + deg["survival_abs_coef"].rank(pct=True)
        + (deg["direction"] == "up_in_gbm").astype(float) * 0.15
    )
    return deg.sort_values("priority", ascending=False)["gene"].dropna().drop_duplicates().head(top_genes).tolist()


def map_genes_to_probes(genes, probemap_path):
    probemap = pd.read_csv(probemap_path, sep="\t")
    probemap = probemap[probemap["gene"].isin(genes)].copy()
    probemap["gene_order"] = probemap["gene"].map({gene: idx for idx, gene in enumerate(genes)})
    probemap["version"] = probemap["id"].str.extract(r"\.(\d+)$").astype(float)
    probemap = probemap.sort_values(["gene_order", "version"], ascending=[True, False])
    return probemap.drop_duplicates("gene")[["gene", "id"]]


def fetch_expression(samples, probes, batch_size):
    chunks = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        _, values = xena_call(DATASET_PROBE_VALUES, XENA_DATASET, samples, batch)
        chunk = pd.DataFrame(values, index=batch, columns=samples).T
        chunks.append(chunk)
        print(f"[{min(start + batch_size, len(probes))}/{len(probes)}] Xena probes fetched")
    return pd.concat(chunks, axis=1)


def differential_expression_from_xena(expression, gbm_samples, normal_samples, gene_map):
    probe_to_gene = dict(zip(gene_map["id"], gene_map["gene"]))
    expression = expression.rename(columns=probe_to_gene)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    gbm = expression.loc[gbm_samples]
    normal = expression.loc[normal_samples]
    genes = expression.columns.tolist()
    t_stat, p_value = stats.ttest_ind(gbm[genes], normal[genes], axis=0, equal_var=False, nan_policy="omit")
    out = pd.DataFrame(
        {
            "gene": genes,
            "gbm_n": int(len(gbm)),
            "control_n": int(len(normal)),
            "gbm_mean_xena_log2_tpm": gbm[genes].mean(axis=0).values,
            "control_mean_xena_log2_tpm": normal[genes].mean(axis=0).values,
            "t_stat": t_stat,
            "p_value": p_value,
        }
    )
    out["log2_fold_change"] = out["gbm_mean_xena_log2_tpm"] - out["control_mean_xena_log2_tpm"]
    out["gbm_mean_tpm"] = out["gbm_mean_xena_log2_tpm"]
    out["control_mean_tpm"] = out["control_mean_xena_log2_tpm"]
    out["fdr_bh"] = bh_fdr(out["p_value"])
    out["direction"] = np.where(out["log2_fold_change"] >= 0, "up_in_gbm", "down_in_gbm")
    out["abs_log2_fold_change"] = out["log2_fold_change"].abs()
    out["deg_score"] = out["abs_log2_fold_change"] * -np.log10(out["fdr_bh"].clip(lower=1e-300))
    out["comparison"] = "TCGA-GBM primary tumor vs GTEx normal brain"
    return out.sort_values(["fdr_bh", "abs_log2_fold_change"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-deg", type=Path, default=Path("results/gbm_deg_proxy.tsv"))
    parser.add_argument("--survival-genes", type=Path, default=Path("results/interpretability_gbm_rna_clinical_fast/rna_gene_coefficients.tsv"))
    parser.add_argument("--phenotype", type=Path, default=Path("data/xena/TcgaTargetGTEX_phenotype.txt.gz"))
    parser.add_argument("--probemap", type=Path, default=Path("data/xena/gencode.v23.annotation.gene.probemap"))
    parser.add_argument("--out-samples", type=Path, default=Path("outputs/xena_gbm_gtex_brain_samples.tsv"))
    parser.add_argument("--out-expression", type=Path, default=Path("data/processed/xena_gbm_gtex_brain_targeted_expression.tsv"))
    parser.add_argument("--out-deg", type=Path, default=Path("results/gbm_vs_gtex_brain_deg_targeted.tsv"))
    parser.add_argument("--out-interactions", type=Path, default=Path("results/dgidb_gbm_normal_targeted_interactions.tsv"))
    parser.add_argument("--out-candidates", type=Path, default=Path("results/drug_repurposing_candidates_gtex_brain.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/drug_repurposing_gtex_brain_summary.json"))
    parser.add_argument("--top-genes", type=int, default=500)
    parser.add_argument("--xena-batch-size", type=int, default=50)
    parser.add_argument("--dgidb-top-genes", type=int, default=140)
    parser.add_argument("--dgidb-batch-size", type=int, default=25)
    parser.add_argument("--dgidb-timeout", type=int, default=60)
    parser.add_argument("--max-gbm-samples", type=int, default=0)
    parser.add_argument("--max-normal-samples", type=int, default=250)
    parser.add_argument("--min-abs-log2-fc", type=float, default=0.8)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--skip-dgidb", action="store_true")
    parser.add_argument("--include-unapproved", action="store_true")
    args = parser.parse_args()

    ensure_download(args.phenotype, PHENOTYPE_URL)
    ensure_download(args.probemap, PROBEMAP_URL)
    phenotype = pd.read_csv(args.phenotype, sep="\t", compression="gzip", encoding="latin1")
    gbm_samples, normal_samples, sample_rows = select_samples(
        phenotype,
        args.max_gbm_samples or None,
        args.max_normal_samples or None,
    )
    args.out_samples.parent.mkdir(parents=True, exist_ok=True)
    sample_rows.to_csv(args.out_samples, sep="\t", index=False)

    genes = prioritized_genes(args.proxy_deg, args.survival_genes, args.top_genes, args.min_abs_log2_fc, args.max_fdr)
    gene_map = map_genes_to_probes(genes, args.probemap)
    samples = gbm_samples + normal_samples
    expression = fetch_expression(samples, gene_map["id"].tolist(), args.xena_batch_size)
    expression = expression.rename(columns=dict(zip(gene_map["id"], gene_map["gene"])))
    expression.insert(0, "sample", expression.index)
    expression.insert(1, "comparison_group", ["GBM primary tumor"] * len(gbm_samples) + ["GTEx normal brain"] * len(normal_samples))
    args.out_expression.parent.mkdir(parents=True, exist_ok=True)
    expression.to_csv(args.out_expression, sep="\t", index=False)

    deg = differential_expression_from_xena(
        expression.drop(columns=["sample", "comparison_group"]),
        gbm_samples,
        normal_samples,
        gene_map,
    )
    survival = load_survival_genes(args.survival_genes)
    deg = deg.merge(survival, on="gene", how="left")
    deg["survival_abs_coef"] = pd.to_numeric(deg["survival_abs_coef"], errors="coerce").fillna(0.0)
    args.out_deg.parent.mkdir(parents=True, exist_ok=True)
    deg.to_csv(args.out_deg, sep="\t", index=False)

    prioritized = deg[
        (deg["fdr_bh"] <= args.max_fdr)
        & (deg["abs_log2_fold_change"] >= args.min_abs_log2_fc)
    ].copy()
    prioritized["priority"] = (
        prioritized["deg_score"].rank(pct=True)
        + prioritized["survival_abs_coef"].rank(pct=True)
        + (prioritized["direction"] == "up_in_gbm").astype(float) * 0.15
    )
    drug_genes = prioritized.sort_values("priority", ascending=False)["gene"].head(args.dgidb_top_genes).tolist()
    interactions = pd.DataFrame()
    if drug_genes and not args.skip_dgidb:
        interactions = query_dgidb(drug_genes, args.dgidb_timeout, args.dgidb_batch_size)
    interactions.to_csv(args.out_interactions, sep="\t", index=False)

    rankable = interactions
    if not args.include_unapproved and not interactions.empty and "drug_approved" in interactions.columns:
        rankable = interactions[interactions["drug_approved"].fillna(False).astype(bool)].copy()
    candidates = rank_candidates(deg, rankable, args.max_candidates)
    candidates["comparison"] = "TCGA-GBM primary tumor vs GTEx normal brain"
    candidates["ranking_note"] = "Targeted normal-brain validation using UCSC Xena Toil TCGA/GTEx expression and DGIdb."
    candidates.to_csv(args.out_candidates, sep="\t", index=False)

    summary = {
        "comparison": "TCGA-GBM primary tumor vs GTEx normal brain",
        "gbm_samples": int(len(gbm_samples)),
        "control_samples": int(len(normal_samples)),
        "genes_requested_from_proxy": int(len(genes)),
        "genes_mapped_to_xena": int(len(gene_map)),
        "genes_tested": int(len(deg)),
        "prioritized_genes": int(len(drug_genes)),
        "dgidb_interactions": int(len(interactions)),
        "rankable_interactions": int(len(rankable)),
        "approved_only": not args.include_unapproved,
        "drug_candidates": int(len(candidates)),
        "out_samples": str(args.out_samples),
        "out_expression": str(args.out_expression),
        "out_deg": str(args.out_deg),
        "out_interactions": str(args.out_interactions),
        "out_candidates": str(args.out_candidates),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
