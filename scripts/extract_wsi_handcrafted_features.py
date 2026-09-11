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
        raise SystemExit("OpenSlide is not installed. Run the WSI dependency setup task first.") from exc
    return openslide


def read_patch(slide, x, y, patch_size, output_size):
    patch = slide.read_region((int(x), int(y)), 0, (int(patch_size), int(patch_size))).convert("RGB")
    if output_size and output_size != patch_size:
        patch = patch.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return np.asarray(patch).astype(np.float32) / 255.0


def patch_features(arr):
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    brightness = arr.mean(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        np.maximum(max_channel, 1e-6),
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    dark = (brightness < 0.45) & (saturation > 0.12)
    purple = (red > 0.25) & (blue > 0.25) & (green < red * 0.92) & (green < blue * 0.98)
    pink = (red > green) & (green > blue) & (brightness > 0.55) & (saturation > 0.08)
    return {
        "red_mean": float(red.mean()),
        "green_mean": float(green.mean()),
        "blue_mean": float(blue.mean()),
        "red_std": float(red.std()),
        "green_std": float(green.std()),
        "blue_std": float(blue.std()),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "dark_pixel_ratio": float(dark.mean()),
        "purple_pixel_ratio": float(purple.mean()),
        "pink_pixel_ratio": float(pink.mean()),
    }


def aggregate_patch_features(patches):
    if not patches:
        return {}
    frame = pd.DataFrame(patches)
    output = {}
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        output[f"wsi_{col}_mean"] = float(values.mean())
        output[f"wsi_{col}_std"] = float(values.std(ddof=0))
        output[f"wsi_{col}_p10"] = float(values.quantile(0.10))
        output[f"wsi_{col}_p90"] = float(values.quantile(0.90))
    return output


def feature_row(slide_row, patch_rows, patches_per_slide, output_size):
    openslide = require_openslide()
    slide_path = Path(slide_row["slide_path"])
    selected = patch_rows.sort_values("tissue_fraction", ascending=False).head(patches_per_slide)
    patch_feature_rows = []

    if slide_path.exists() and len(selected):
        slide = openslide.OpenSlide(str(slide_path))
        try:
            for _, patch in selected.iterrows():
                arr = read_patch(slide, patch["x"], patch["y"], patch["patch_size"], output_size)
                row = patch_features(arr)
                row["tissue_fraction"] = float(patch["tissue_fraction"])
                patch_feature_rows.append(row)
        finally:
            slide.close()

    width = float(slide_row.get("width", 0) or 0)
    height = float(slide_row.get("height", 0) or 0)
    tissue_fraction = float(slide_row.get("tissue_fraction_thumbnail", 0) or 0)
    row = {
        "case_submitter_id": slide_row["case_submitter_id"],
        "project_id": slide_row.get("project_id", ""),
        "file_id": slide_row["file_id"],
        "file_name": slide_row.get("file_name", ""),
        "wsi_width": width,
        "wsi_height": height,
        "wsi_area_megapixels": width * height / 1_000_000,
        "wsi_tissue_fraction_thumbnail": tissue_fraction,
        "wsi_tissue_area_megapixels": width * height * tissue_fraction / 1_000_000,
        "wsi_candidate_patch_count": int(slide_row.get("patches", 0) or 0),
        "wsi_patches_used": len(patch_feature_rows),
    }
    row.update(aggregate_patch_features(patch_feature_rows))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-summary", type=Path, default=Path("data/processed/wsi_smoke/slide_summary.tsv"))
    parser.add_argument("--patch-index", type=Path, default=Path("data/processed/wsi_smoke/patch_index.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/wsi_smoke/wsi_handcrafted_features.tsv"))
    parser.add_argument("--patches-per-slide", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()

    slides = pd.read_csv(args.slide_summary, sep="\t")
    patches = pd.read_csv(args.patch_index, sep="\t") if args.patch_index.exists() else pd.DataFrame()
    rows = []
    for _, slide in slides.iterrows():
        slide_patches = patches[patches["file_id"] == slide["file_id"]] if not patches.empty else pd.DataFrame()
        rows.append(feature_row(slide, slide_patches, args.patches_per_slide, args.output_size))

    features = pd.DataFrame(rows).fillna(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, sep="\t", index=False)
    print(f"wrote WSI handcrafted features: {args.out} ({len(features)} slides, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
