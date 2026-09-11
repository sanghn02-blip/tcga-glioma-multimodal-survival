#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-file-map", type=Path, default=Path("outputs/gdc_wsi_file_map_primary_overlap.tsv"))
    parser.add_argument("--wsi-manifest", type=Path, default=Path("outputs/gdc_manifest_wsi_diagnostic_primary_overlap.tsv"))
    parser.add_argument("--cases", type=Path, default=Path("outputs/tcga_lgg_gbm_primary_overlap_cases.tsv"))
    parser.add_argument("--out-manifest", type=Path, default=Path("outputs/gdc_manifest_wsi_smoke.tsv"))
    parser.add_argument("--out-file-map", type=Path, default=Path("outputs/gdc_wsi_file_map_smoke.tsv"))
    parser.add_argument("--out-cases", type=Path, default=Path("outputs/tcga_lgg_gbm_wsi_smoke_cases.tsv"))
    parser.add_argument("--n-per-project", type=int, default=3)
    parser.add_argument("--max-size-mb", type=float, default=100.0)
    args = parser.parse_args()

    file_map = pd.read_csv(args.wsi_file_map, sep="\t")
    manifest = pd.read_csv(args.wsi_manifest, sep="\t")
    cases = pd.read_csv(args.cases, sep="\t")
    max_size = args.max_size_mb * 1024 * 1024

    selected = []
    candidates = file_map[file_map["file_size"].le(max_size)].sort_values(["project_id", "file_size", "case_submitter_id"])
    for project_id, group in candidates.groupby("project_id", sort=True):
        seen_cases = set()
        for _, row in group.iterrows():
            if row["case_submitter_id"] in seen_cases:
                continue
            selected.append(row)
            seen_cases.add(row["case_submitter_id"])
            if len(seen_cases) >= args.n_per_project:
                break

    selected_map = pd.DataFrame(selected)
    if selected_map.empty:
        raise ValueError("No WSI files matched the smoke manifest constraints.")

    selected_manifest = manifest[manifest["id"].isin(selected_map["file_id"])].copy()
    selected_cases = cases[cases["case_submitter_id"].isin(selected_map["case_submitter_id"])].copy()

    for path in [args.out_manifest, args.out_file_map, args.out_cases]:
        path.parent.mkdir(parents=True, exist_ok=True)

    selected_manifest.to_csv(args.out_manifest, sep="\t", index=False)
    selected_map.to_csv(args.out_file_map, sep="\t", index=False)
    selected_cases.to_csv(args.out_cases, sep="\t", index=False)

    total_mb = selected_map["file_size"].sum() / 1024 / 1024
    print(f"wrote {len(selected_map)} WSI smoke files ({total_mb:.1f} MB)")
    print(f"manifest: {args.out_manifest}")
    print(f"file map: {args.out_file_map}")
    print(f"cases: {args.out_cases}")
    print(selected_map[["case_submitter_id", "project_id", "file_name", "file_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
