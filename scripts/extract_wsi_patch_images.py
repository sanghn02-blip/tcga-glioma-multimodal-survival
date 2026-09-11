#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def require_openslide():
    try:
        import openslide
    except ModuleNotFoundError as exc:
        message = (
            "OpenSlide Python is not installed. Run:\n"
            "  .venv/bin/python -m pip install -r requirements-wsi.txt\n"
        )
        raise SystemExit(message) from exc
    return openslide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-index", type=Path, default=Path("data/processed/wsi_smoke/patch_index.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wsi_smoke/patch_images"))
    parser.add_argument("--patches-per-slide", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()

    openslide = require_openslide()
    patch_index = pd.read_csv(args.patch_index, sep="\t")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for file_id, group in patch_index.groupby("file_id", sort=True):
        slide_path = Path(group.iloc[0]["slide_path"])
        slide = openslide.OpenSlide(str(slide_path))
        selected = group.sort_values("tissue_fraction", ascending=False).head(args.patches_per_slide)
        slide_out_dir = args.out_dir / file_id
        slide_out_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            x = int(row["x"])
            y = int(row["y"])
            patch_size = int(row["patch_size"])
            image = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            if args.output_size and args.output_size != patch_size:
                image = image.resize((args.output_size, args.output_size))
            out_path = slide_out_dir / f"patch_{rank:03d}_x{x}_y{y}.jpg"
            image.save(out_path, quality=92)
            rows.append(
                {
                    "case_submitter_id": row["case_submitter_id"],
                    "project_id": row["project_id"],
                    "file_id": file_id,
                    "file_name": row["file_name"],
                    "rank": rank,
                    "x": x,
                    "y": y,
                    "patch_size": patch_size,
                    "output_size": args.output_size,
                    "tissue_fraction": row["tissue_fraction"],
                    "patch_image_path": str(out_path),
                }
            )
        slide.close()

    pd.DataFrame(rows).to_csv(args.out_dir / "patch_image_index.tsv", sep="\t", index=False)
    print(f"wrote {len(rows)} patch images")
    print(f"index: {args.out_dir / 'patch_image_index.tsv'}")


if __name__ == "__main__":
    main()
