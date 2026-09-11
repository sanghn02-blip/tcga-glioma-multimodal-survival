#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd


def find_downloaded_file(root, file_id, file_name):
    candidates = [
        root / file_id / file_name,
        root / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(file_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {file_name} for file_id={file_id} under {root}")


def read_star_counts(path, value_column):
    frame = pd.read_csv(path, sep="\t", comment="#")
    if value_column not in frame.columns:
        raise ValueError(f"{path} does not contain column {value_column}; columns={list(frame.columns)}")
    frame = frame[frame["gene_type"].eq("protein_coding")].copy()
    frame = frame[~frame["gene_id"].astype(str).str.startswith("N_")]
    series = frame.set_index("gene_name")[value_column]
    series = pd.to_numeric(series, errors="coerce")
    return series.groupby(level=0).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-dir", required=True, type=Path)
    parser.add_argument("--file-map", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--value-column", default="tpm_unstranded")
    args = parser.parse_args()

    case_order = pd.read_csv(args.cases, sep="\t")["case_submitter_id"].tolist()
    file_map = pd.read_csv(args.file_map, sep="\t")

    columns = {}
    for _, row in file_map.iterrows():
        case_id = row["case_submitter_id"]
        if case_id not in case_order:
            continue
        path = find_downloaded_file(args.rna_dir, row["file_id"], row["file_name"])
        columns.setdefault(case_id, []).append(read_star_counts(path, args.value_column))

    collapsed = {}
    for case_id, series_list in columns.items():
        collapsed[case_id] = pd.concat(series_list, axis=1).mean(axis=1)

    matrix = pd.DataFrame(collapsed).T
    matrix = matrix.reindex(case_order).dropna(how="all")
    matrix.index.name = "case_submitter_id"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out, sep="\t")
    print(f"wrote {matrix.shape[0]} cases x {matrix.shape[1]} genes to {args.out}")


if __name__ == "__main__":
    main()
