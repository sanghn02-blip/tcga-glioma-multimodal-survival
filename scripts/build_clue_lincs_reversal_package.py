#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from run_gbm_drug_repurposing import ACTIVATING_TERMS, INHIBITORY_TERMS


CLUE_JOBS_URL = "https://api.clue.io/api/jobs"
MYGENE_QUERY_URL = "https://mygene.info/v3/query"


def clean_gene_list(values, limit):
    genes = []
    seen = set()
    for value in values:
        gene = str(value).strip()
        if not gene or gene.lower() == "nan" or gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
        if len(genes) >= limit:
            break
    return genes


def write_gmt(path, name, genes, description):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join([name, description] + genes) + "\n", encoding="utf-8")


def map_symbols_to_entrez(symbols, out_path, timeout):
    response = requests.post(
        MYGENE_QUERY_URL,
        data={
            "q": ",".join(symbols),
            "scopes": "symbol",
            "fields": "entrezgene,symbol",
            "species": "human",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json()
    symbol_to_entrez = {}
    mapping_rows = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        entrez = row.get("entrezgene")
        symbol = row.get("symbol")
        if query and entrez is not None and query not in symbol_to_entrez:
            symbol_to_entrez[query] = str(entrez)
        mapping_rows.append(
            {
                "query_symbol": query,
                "matched_symbol": symbol,
                "entrezgene": entrez,
                "notfound": row.get("notfound", False),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(out_path, sep="\t", index=False)
    return symbol_to_entrez


def select_query_genes(deg, up_n, down_n):
    ranked = deg.copy()
    ranked["abs_log2_fold_change"] = pd.to_numeric(ranked["abs_log2_fold_change"], errors="coerce").fillna(0)
    ranked["fdr_bh"] = pd.to_numeric(ranked["fdr_bh"], errors="coerce").fillna(1)
    ranked = ranked.sort_values(["fdr_bh", "abs_log2_fold_change"], ascending=[True, False])
    up = clean_gene_list(ranked[ranked["direction"].eq("up_in_gbm")]["gene"], up_n)
    down = clean_gene_list(ranked[ranked["direction"].eq("down_in_gbm")]["gene"], down_n)
    return up, down


def term_match(types, terms):
    text = str(types).lower()
    return any(term in text for term in terms)


def candidate_reversal_prior(candidate):
    direction = str(candidate.get("primary_target_direction", ""))
    types = str(candidate.get("interaction_types", "")).lower()
    inhibitory = term_match(types, INHIBITORY_TERMS)
    activating = term_match(types, ACTIVATING_TERMS)
    if direction == "up_in_gbm" and inhibitory:
        return 1.0, "target_inhibits_gbm_up_gene"
    if direction == "down_in_gbm" and activating:
        return 1.0, "target_activates_gbm_down_gene"
    if inhibitory or activating:
        return 0.45, "directional_mechanism_known_but_not_reversal"
    return 0.35, "mechanism_ambiguous"


def minmax(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low, high = values.min(), values.max()
    if high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def build_clue_ready_ranking(candidates):
    out = candidates.copy()
    priors = out.apply(candidate_reversal_prior, axis=1)
    out["local_reversal_prior"] = [score for score, _ in priors]
    out["local_reversal_reason"] = [reason for _, reason in priors]
    out["target_count_norm"] = minmax(out["target_count"])
    out["repurposing_score_norm"] = minmax(out["repurposing_score"])
    out["clue_tau_score"] = np.nan
    out["clue_connectivity_direction"] = ""
    out["clue_status"] = "pending_api_key"
    out["final_repurposing_score"] = (
        0.72 * out["repurposing_score_norm"]
        + 0.20 * out["local_reversal_prior"]
        + 0.08 * out["target_count_norm"]
    )
    out = out.sort_values("final_repurposing_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.drop(columns=["target_count_norm", "repurposing_score_norm"])


def submit_clue_job(up_gmt, down_gmt, name, api_key, timeout):
    response = requests.post(
        CLUE_JOBS_URL,
        headers={"user_key": api_key, "Accept": "application/json"},
        json={
            "tool_id": "sig_gutc_tool",
            "name": name,
            "data_type": "L1000",
            "dataset": "Touchstone",
            "ignoreWarnings": True,
            "uptag-cmapfile": up_gmt.read_text(encoding="utf-8").strip(),
            "dntag-cmapfile": down_gmt.read_text(encoding="utf-8").strip(),
        },
        timeout=timeout,
    )
    payload = {"status_code": response.status_code}
    try:
        payload["response"] = response.json()
    except ValueError:
        payload["response_text"] = response.text
    response.raise_for_status()
    if isinstance(payload.get("response"), dict):
        params = payload["response"].get("result", {}).get("params", {})
        params.pop("api_key", None)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deg", type=Path, default=Path("results/gbm_vs_gtex_brain_deg_targeted.tsv"))
    parser.add_argument("--candidates", type=Path, default=Path("results/drug_repurposing_candidates_gtex_brain.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/clue_lincs"))
    parser.add_argument("--out-candidates", type=Path, default=Path("results/drug_repurposing_candidates_clue_ready.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/clue_lincs_reversal_summary.json"))
    parser.add_argument("--up-genes", type=int, default=150)
    parser.add_argument("--down-genes", type=int, default=150)
    parser.add_argument("--clue-feature-id", choices=["symbol", "entrez"], default="entrez")
    parser.add_argument("--submit-clue", action="store_true")
    parser.add_argument("--clue-api-key-env", default="CLUE_API_KEY")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    deg = pd.read_csv(args.deg, sep="\t")
    candidates = pd.read_csv(args.candidates, sep="\t")
    up, down = select_query_genes(deg, args.up_genes, args.down_genes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    symbol_up_gmt = args.out_dir / "gbm_vs_gtex_brain_up.gmt"
    symbol_down_gmt = args.out_dir / "gbm_vs_gtex_brain_down.gmt"
    write_gmt(symbol_up_gmt, "GBM_VS_GTEX_BRAIN_UP", up, "Genes higher in TCGA GBM primary tumor than GTEx normal brain")
    write_gmt(symbol_down_gmt, "GBM_VS_GTEX_BRAIN_DN", down, "Genes lower in TCGA GBM primary tumor than GTEx normal brain")

    up_gmt = symbol_up_gmt
    down_gmt = symbol_down_gmt
    clue_feature_id = "symbol"
    mapped_up = up
    mapped_down = down
    mapping_path = args.out_dir / "gbm_vs_gtex_brain_gene_symbol_to_entrez.tsv"
    if args.clue_feature_id == "entrez":
        symbol_to_entrez = map_symbols_to_entrez(up + down, mapping_path, args.timeout)
        mapped_up = [symbol_to_entrez[symbol] for symbol in up if symbol in symbol_to_entrez]
        mapped_down = [symbol_to_entrez[symbol] for symbol in down if symbol in symbol_to_entrez]
        up_gmt = args.out_dir / "gbm_vs_gtex_brain_up_entrez.gmt"
        down_gmt = args.out_dir / "gbm_vs_gtex_brain_down_entrez.gmt"
        write_gmt(up_gmt, "GBM_VS_GTEX_BRAIN_UP", mapped_up, "Entrez IDs for genes higher in TCGA GBM primary tumor than GTEx normal brain")
        write_gmt(down_gmt, "GBM_VS_GTEX_BRAIN_DN", mapped_down, "Entrez IDs for genes lower in TCGA GBM primary tumor than GTEx normal brain")
        clue_feature_id = "entrez"

    clue_response = None
    clue_status = "not_submitted_requires_clue_api_key"
    if args.submit_clue:
        api_key = os.environ.get(args.clue_api_key_env)
        if not api_key:
            clue_status = f"not_submitted_missing_env_{args.clue_api_key_env}"
        else:
            clue_response = submit_clue_job(up_gmt, down_gmt, "GBM_vs_GTEx_brain_reversal", api_key, args.timeout)
            clue_status = "submitted"

    ranked = build_clue_ready_ranking(candidates)
    ranked["clue_query_up_gmt"] = str(up_gmt)
    ranked["clue_query_down_gmt"] = str(down_gmt)
    ranked["clue_status"] = "submitted" if clue_status == "submitted" else "pending_api_key"
    args.out_candidates.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out_candidates, sep="\t", index=False)

    summary = {
        "clue_status": clue_status,
        "comparison": "TCGA-GBM primary tumor vs GTEx normal brain",
        "up_genes": len(up),
        "down_genes": len(down),
        "clue_feature_id": clue_feature_id,
        "clue_up_features": len(mapped_up),
        "clue_down_features": len(mapped_down),
        "candidate_drugs": int(len(ranked)),
        "top_candidate": ranked.iloc[0]["drug_name"] if not ranked.empty else None,
        "top_candidate_target": ranked.iloc[0]["primary_target"] if not ranked.empty else None,
        "up_gmt": str(up_gmt),
        "down_gmt": str(down_gmt),
        "symbol_up_gmt": str(symbol_up_gmt),
        "symbol_down_gmt": str(symbol_down_gmt),
        "gene_id_mapping": str(mapping_path) if args.clue_feature_id == "entrez" else None,
        "out_candidates": str(args.out_candidates),
        "note": "CLUE API submission requires a registered clue.io API key. This file is CLUE-ready and stores local reversal-prior scores until tau scores are available.",
    }
    if clue_response is not None:
        response = clue_response.get("response", {}) if isinstance(clue_response, dict) else {}
        result = response.get("result", {}) if isinstance(response, dict) else {}
        summary["clue_job_id"] = result.get("job_id")
        summary["clue_response_status"] = response.get("status")
        summary["clue_response_status_code"] = clue_response.get("status_code")
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
