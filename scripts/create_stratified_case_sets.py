#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def clean_label(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def write_case_set(name, frame, out_dir, rows):
    path = out_dir / f"{name}.tsv"
    frame.to_csv(path, sep="\t", index=False)
    events = int(pd.to_numeric(frame["os_event"], errors="coerce").fillna(0).sum())
    rows.append(
        {
            "case_set": name,
            "n_cases": len(frame),
            "n_events": events,
            "path": str(path),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--cbio", type=Path, default=Path("outputs/cbioportal_pancan_glioma_covariates.tsv"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-cases", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.cases, sep="\t")
    cbio = pd.read_csv(args.cbio, sep="\t")

    merged = cases.merge(cbio, on="case_submitter_id", how="left")
    merged["os_days"] = pd.to_numeric(merged["os_days"], errors="coerce")
    merged["os_event"] = pd.to_numeric(merged["os_event"], errors="coerce")
    merged = merged[merged["os_days"].notna() & merged["os_event"].notna() & merged["os_days"].gt(0)].copy()

    case_columns = cases.columns.tolist()
    rows = []

    for project_id, group in merged.groupby("project_id", dropna=True):
        name = clean_label(project_id)
        if len(group) >= args.min_cases:
            write_case_set(name, group[case_columns], args.out_dir, rows)

    for subtype, group in merged.groupby("pancan_subtype", dropna=True):
        if not str(subtype).strip():
            continue
        name = clean_label(subtype)
        if len(group) >= args.min_cases:
            write_case_set(name, group[case_columns], args.out_dir, rows)

    merged["grade_for_strata"] = merged["tumor_grade"].fillna(merged["histologic_grade"])
    for grade, group in merged.groupby("grade_for_strata", dropna=True):
        if not str(grade).strip():
            continue
        name = f"grade_{clean_label(grade)}"
        if len(group) >= args.min_cases:
            write_case_set(name, group[case_columns], args.out_dir, rows)

    summary = pd.DataFrame(rows).sort_values(["n_cases", "case_set"], ascending=[False, True])
    summary.to_csv(args.out_dir / "case_set_summary.tsv", sep="\t", index=False)
    print(f"wrote {len(summary)} case sets to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
