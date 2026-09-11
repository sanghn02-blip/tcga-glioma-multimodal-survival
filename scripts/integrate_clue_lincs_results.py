#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CLUE_PERTS_URL = "https://api.clue.io/api/perts"
SALT_SUFFIXES = [
    " hydrochloride",
    " hcl",
    " sodium",
    " phosphate",
    " sulfate",
    " acetate",
]


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".gct":
        lines = path.read_text(errors="ignore").splitlines()
        start = 0
        if lines and lines[0].startswith("#"):
            start = 2 if len(lines) > 2 else 1
        return pd.read_csv(path, sep="\t", skiprows=start)
    return pd.DataFrame()


def norm(value):
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def query_names(names):
    out = []
    for name in names:
        clean = norm(name)
        if clean:
            out.append(clean)
        for suffix in SALT_SUFFIXES:
            if clean.endswith(suffix.strip()):
                out.append(clean[: -len(suffix.strip())].strip())
    seen = set()
    unique = []
    for name in out:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def fetch_clue_pert_metadata(candidate_names, api_key, timeout, out_path):
    names = query_names(candidate_names)
    records = []
    for start in range(0, len(names), 40):
        chunk = names[start : start + 40]
        query_filter = {
            "fields": ["pert_id", "pert_iname", "pert_type", "moa", "target"],
            "where": {"pert_iname": {"inq": chunk}},
        }
        response = requests.get(
            CLUE_PERTS_URL,
            headers={"user_key": api_key, "Accept": "application/json"},
            params={"filter": json.dumps(query_filter)},
            timeout=timeout,
        )
        response.raise_for_status()
        records.extend(response.json())
    frame = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, sep="\t", index=False)
    return frame


def candidate_aliases(candidates):
    rows = []
    for name in candidates["drug_name"].dropna().unique():
        rows.append({"alias": norm(name), "drug_name": name})
        for alias in query_names([name]):
            rows.append({"alias": alias, "drug_name": name})
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates()


def collect_pert_id_scores(results_dir):
    rows = []
    for path in results_dir.rglob("pert_id_summary.gct"):
        try:
            frame = read_table(path)
        except Exception:
            continue
        if frame.empty or "id" not in frame.columns:
            continue
        score_cols = [col for col in frame.columns if col != "id"]
        if not score_cols:
            continue
        for _, row in frame.iterrows():
            score = pd.to_numeric(pd.Series([row.get(score_cols[0])]), errors="coerce").iloc[0]
            if pd.isna(score):
                continue
            rows.append(
                {
                    "pert_id": row.get("id"),
                    "clue_tau_score": float(score),
                    "clue_source_file": str(path),
                    "clue_score_column": score_cols[0],
                }
            )
    return pd.DataFrame(rows)


def collect_candidate_scores(results_dir, candidates, pert_metadata):
    aliases = candidate_aliases(candidates)
    pert_scores = collect_pert_id_scores(results_dir)
    if pert_scores.empty or pert_metadata.empty:
        return pd.DataFrame()
    metadata = pert_metadata.copy()
    metadata["alias"] = metadata["pert_iname"].map(norm)
    metadata = metadata.merge(aliases, on="alias", how="inner")
    scores = pert_scores.merge(metadata, on="pert_id", how="inner")
    if scores.empty:
        return pd.DataFrame()
    scores["reversal_strength"] = -pd.to_numeric(scores["clue_tau_score"], errors="coerce")
    scores = scores.sort_values("reversal_strength", ascending=False).drop_duplicates("drug_name")
    scores = scores.rename(columns={"pert_iname": "clue_pert_iname"})
    return scores[
        [
            "drug_name",
            "pert_id",
            "clue_pert_iname",
            "pert_type",
            "moa",
            "target",
            "clue_tau_score",
            "clue_source_file",
            "clue_score_column",
        ]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/clue_lincs/clue_lincs_results"))
    parser.add_argument("--candidates", type=Path, default=Path("results/drug_repurposing_candidates_clue_ready.tsv"))
    parser.add_argument("--validation", type=Path, default=Path("results/drug_repurposing_validation_matrix.tsv"))
    parser.add_argument("--out-candidates", type=Path, default=Path("results/drug_repurposing_candidates_with_clue_tau.tsv"))
    parser.add_argument("--out-validation", type=Path, default=Path("results/drug_repurposing_validation_matrix_with_clue_tau.tsv"))
    parser.add_argument("--out-pert-metadata", type=Path, default=Path("results/clue_lincs/clue_candidate_pert_metadata.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/clue_lincs_integration_summary.json"))
    parser.add_argument("--clue-summary", type=Path, default=Path("results/clue_lincs_reversal_summary.json"))
    parser.add_argument("--clue-api-key-env", default="CLUE_API_KEY")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, sep="\t")
    api_key = os.environ.get(args.clue_api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {args.clue_api_key_env}. Set it before integrating CLUE perturbagen metadata.")
    pert_metadata = fetch_clue_pert_metadata(candidates["drug_name"], api_key, args.timeout, args.out_pert_metadata)
    scores = collect_candidate_scores(args.results_dir, candidates, pert_metadata)
    if scores.empty:
        summary = {
            "status": "no_candidate_scores_found",
            "results_dir": str(args.results_dir),
            "candidate_drugs": int(len(candidates)),
            "note": "CLUE results or CLUE perturbagen metadata did not match the candidate drug names. Inspect pert_id_summary.gct and clue_candidate_pert_metadata.tsv.",
        }
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    merged = candidates.drop(columns=["clue_tau_score"], errors="ignore").merge(scores, on="drug_name", how="left")
    merged["clue_connectivity_direction"] = np.where(
        pd.to_numeric(merged["clue_tau_score"], errors="coerce") < 0,
        "reversal",
        np.where(pd.to_numeric(merged["clue_tau_score"], errors="coerce") > 0, "mimicry", ""),
    )
    merged["clue_status"] = np.where(merged["clue_tau_score"].notna(), "completed", merged.get("clue_status", "completed"))
    clue_bonus = (-pd.to_numeric(merged["clue_tau_score"], errors="coerce").fillna(0)).clip(lower=0) / 100.0
    merged["final_repurposing_score"] = (
        pd.to_numeric(merged["final_repurposing_score"], errors="coerce").fillna(0) + 0.20 * clue_bonus
    ).clip(upper=1.0)
    merged = merged.sort_values("final_repurposing_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    merged.to_csv(args.out_candidates, sep="\t", index=False)

    validation_out = None
    if args.validation.exists():
        validation = pd.read_csv(args.validation, sep="\t")
        validation = validation.drop(columns=["clue_tau_score", "clue_source_file"], errors="ignore")
        validation = validation.merge(scores[["drug_name", "pert_id", "clue_pert_iname", "clue_tau_score", "clue_source_file"]], on="drug_name", how="left")
        clue_bonus = (-pd.to_numeric(validation["clue_tau_score"], errors="coerce").fillna(0)).clip(lower=0) / 100.0
        validation["validation_score"] = (
            pd.to_numeric(validation["validation_score"], errors="coerce").fillna(0) + 0.20 * clue_bonus
        ).clip(upper=1.0)
        validation = validation.sort_values("validation_score", ascending=False).reset_index(drop=True)
        validation["validation_rank"] = np.arange(1, len(validation) + 1)
        validation.to_csv(args.out_validation, sep="\t", index=False)
        validation_out = str(args.out_validation)

    summary = {
        "status": "integrated",
        "candidate_scores_found": int(len(scores)),
        "top_candidate": merged.iloc[0]["drug_name"] if not merged.empty else None,
        "top_candidate_tau": float(merged.iloc[0]["clue_tau_score"]) if pd.notna(merged.iloc[0].get("clue_tau_score")) else None,
        "metadata_rows": int(len(pert_metadata)),
        "out_pert_metadata": str(args.out_pert_metadata),
        "out_candidates": str(args.out_candidates),
        "out_validation": validation_out,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.clue_summary.exists():
        clue_summary = json.loads(args.clue_summary.read_text(encoding="utf-8"))
        clue_summary.update(
            {
                "clue_status": "completed",
                "clue_tau_integrated": True,
                "clue_candidate_scores_found": int(len(scores)),
                "clue_top_candidate_tau": summary["top_candidate_tau"],
                "out_candidates_with_tau": str(args.out_candidates),
                "out_validation_with_tau": validation_out,
            }
        )
        args.clue_summary.write_text(json.dumps(clue_summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
