#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def require_openslide():
    try:
        import openslide
    except ModuleNotFoundError as exc:
        message = (
            "OpenSlide Python is not installed. On macOS, run:\n"
            "  brew install openslide\n"
            "  .venv/bin/python -m pip install openslide-python\n"
        )
        raise SystemExit(message) from exc
    return openslide


def locate_slide(root, file_id, file_name):
    candidates = [root / file_id / file_name, root / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(file_name)) if root.exists() else []
    return matches[0] if matches else None


def tissue_mask_from_thumbnail(image, saturation_threshold, brightness_min, brightness_max):
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        np.maximum(max_channel, 1e-6),
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )
    brightness = arr.mean(axis=2)
    return (saturation >= saturation_threshold) & (brightness >= brightness_min) & (brightness <= brightness_max)


def save_mask(mask, out_path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def save_overlay(thumbnail, mask, out_path):
    base = thumbnail.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    red = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    red[mask] = [204, 69, 54, 95]
    overlay = Image.fromarray(red, mode="RGBA")
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path, quality=90)


def build_patch_rows(row, slide_path, slide, mask, thumb_size, patch_size, stride, min_tissue_fraction, max_patches):
    width, height = slide.dimensions
    thumb_w, thumb_h = thumb_size
    rows = []
    for y in range(0, max(height - patch_size + 1, 1), stride):
        for x in range(0, max(width - patch_size + 1, 1), stride):
            tx0 = int(x / width * thumb_w)
            ty0 = int(y / height * thumb_h)
            tx1 = max(tx0 + 1, int((x + patch_size) / width * thumb_w))
            ty1 = max(ty0 + 1, int((y + patch_size) / height * thumb_h))
            tx1 = min(tx1, thumb_w)
            ty1 = min(ty1, thumb_h)
            tissue_fraction = float(mask[ty0:ty1, tx0:tx1].mean()) if tx1 > tx0 and ty1 > ty0 else 0.0
            if tissue_fraction < min_tissue_fraction:
                continue
            rows.append(
                {
                    "case_submitter_id": row["case_submitter_id"],
                    "project_id": row["project_id"],
                    "file_id": row["file_id"],
                    "file_name": row["file_name"],
                    "slide_path": str(slide_path),
                    "level": 0,
                    "x": x,
                    "y": y,
                    "patch_size": patch_size,
                    "stride": stride,
                    "tissue_fraction": tissue_fraction,
                }
            )
    rows = sorted(rows, key=lambda item: item["tissue_fraction"], reverse=True)
    return rows[:max_patches]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-dir", type=Path, default=Path("data/gdc/wsi_smoke"))
    parser.add_argument("--file-map", type=Path, default=Path("outputs/gdc_wsi_file_map_smoke.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wsi_smoke"))
    parser.add_argument("--thumbnail-size", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.50)
    parser.add_argument("--max-patches-per-slide", type=int, default=200)
    parser.add_argument("--saturation-threshold", type=float, default=0.08)
    parser.add_argument("--brightness-min", type=float, default=0.18)
    parser.add_argument("--brightness-max", type=float, default=0.92)
    args = parser.parse_args()

    openslide = require_openslide()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    file_map = pd.read_csv(args.file_map, sep="\t")
    all_rows = []
    slide_rows = []

    for _, row in file_map.iterrows():
        slide_path = locate_slide(args.wsi_dir, row["file_id"], row["file_name"])
        if slide_path is None:
            slide_rows.append(
                {
                    "case_submitter_id": row["case_submitter_id"],
                    "file_id": row["file_id"],
                    "file_name": row["file_name"],
                    "status": "missing",
                    "patches": 0,
                }
            )
            continue

        slide = openslide.OpenSlide(str(slide_path))
        width, height = slide.dimensions
        scale = args.thumbnail_size / max(width, height)
        thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
        mask = tissue_mask_from_thumbnail(
            thumbnail,
            args.saturation_threshold,
            args.brightness_min,
            args.brightness_max,
        )

        stem = row["file_id"]
        slide_out_dir = args.out_dir / stem
        slide_out_dir.mkdir(parents=True, exist_ok=True)
        thumbnail.save(slide_out_dir / "thumbnail.jpg", quality=90)
        save_mask(mask, slide_out_dir / "tissue_mask.png")
        save_overlay(thumbnail, mask, slide_out_dir / "tissue_overlay.jpg")

        patch_rows = build_patch_rows(
            row,
            slide_path,
            slide,
            mask,
            thumbnail.size,
            args.patch_size,
            args.stride,
            args.min_tissue_fraction,
            args.max_patches_per_slide,
        )
        pd.DataFrame(patch_rows).to_csv(slide_out_dir / "patch_index.tsv", sep="\t", index=False)
        all_rows.extend(patch_rows)
        slide_rows.append(
            {
                "case_submitter_id": row["case_submitter_id"],
                "project_id": row["project_id"],
                "file_id": row["file_id"],
                "file_name": row["file_name"],
                "slide_path": str(slide_path),
                "width": width,
                "height": height,
                "thumbnail_width": thumbnail.size[0],
                "thumbnail_height": thumbnail.size[1],
                "tissue_fraction_thumbnail": float(mask.mean()),
                "patches": len(patch_rows),
                "status": "processed",
                "thumbnail_path": str(slide_out_dir / "thumbnail.jpg"),
                "overlay_path": str(slide_out_dir / "tissue_overlay.jpg"),
            }
        )
        slide.close()

    pd.DataFrame(slide_rows).to_csv(args.out_dir / "slide_summary.tsv", sep="\t", index=False)
    pd.DataFrame(all_rows).to_csv(args.out_dir / "patch_index.tsv", sep="\t", index=False)
    print(f"processed slides: {sum(row['status'] == 'processed' for row in slide_rows)} / {len(slide_rows)}")
    print(f"patches: {len(all_rows)}")
    print(f"summary: {args.out_dir / 'slide_summary.tsv'}")
    print(f"patch index: {args.out_dir / 'patch_index.tsv'}")


if __name__ == "__main__":
    main()

