#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def draw_label(draw, x, y, text, fill):
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    bg = (30, 32, 38, 210)
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=4,
        fill=bg,
    )
    draw.text((x, y), text, fill=fill, font=font)


def draw_legend(draw, width, height):
    font = ImageFont.load_default()
    items = [
        ("Raises risk", (220, 62, 48, 235)),
        ("Lowers risk", (38, 99, 235, 235)),
    ]
    x = 14
    y = max(14, height - 54)
    box_w = 184
    box_h = 40
    draw.rounded_rectangle((x - 8, y - 8, x + box_w, y + box_h), radius=6, fill=(255, 255, 255, 215))
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 18
        draw.rectangle((x, yy + 2, x + 12, yy + 12), outline=color, width=3)
        draw.text((x + 20, yy), label, fill=(30, 32, 38, 235), font=font)


def overlay_for_slide(slide, patches, out_path, base_column, top_n):
    base_path = Path(slide[base_column])
    if not base_path.exists():
        return None

    width = as_float(slide.get("width"))
    height = as_float(slide.get("height"))
    if width <= 0 or height <= 0:
        return None

    base = Image.open(base_path).convert("RGBA")
    draw_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    thumb_w, thumb_h = base.size

    patches = patches.sort_values(["patch_risk_rank", "rank"], na_position="last").head(top_n)
    for _, patch in patches.iterrows():
        x = as_float(patch.get("x"))
        y = as_float(patch.get("y"))
        patch_size = as_float(patch.get("patch_size"), 512)
        x0 = int(x / width * thumb_w)
        y0 = int(y / height * thumb_h)
        x1 = max(x0 + 2, int((x + patch_size) / width * thumb_w))
        y1 = max(y0 + 2, int((y + patch_size) / height * thumb_h))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(thumb_w - 1, x1), min(thumb_h - 1, y1)

        direction = str(patch.get("patch_risk_direction", ""))
        color = (220, 62, 48, 230) if direction == "raises_risk" else (38, 99, 235, 220)
        norm = max(0.0, min(1.0, as_float(patch.get("patch_importance_norm"))))
        line_width = 2 + int(round(norm * 4))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)

        label = f"R{as_int(patch.get('patch_risk_rank'), as_int(patch.get('rank')))}"
        draw_label(draw, x0 + 4, max(4, y0 - 15), label, (255, 255, 255, 255))

    draw_legend(draw, thumb_w, thumb_h)
    merged = Image.alpha_composite(base, draw_layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, quality=92)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-summary", type=Path, default=Path("data/processed/wsi_dev_50/slide_summary.tsv"))
    parser.add_argument("--patch-importance", type=Path, default=Path("results/wsi_uni_patch_importance.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/wsi_uni_risk_overlay_summary.tsv"))
    parser.add_argument("--base-column", default="overlay_path", choices=["overlay_path", "thumbnail_path"])
    parser.add_argument("--image-name", default="risk_overlay_uni.jpg")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    slides = pd.read_csv(args.slide_summary, sep="\t")
    patches = pd.read_csv(args.patch_importance, sep="\t")
    required = {"file_id", "rank", "x", "y", "patch_size", "patch_risk_rank"}
    missing = required.difference(patches.columns)
    if missing:
        raise ValueError(f"Patch importance file is missing columns: {', '.join(sorted(missing))}")

    rows = []
    for _, slide in slides.iterrows():
        slide_patches = patches[patches["file_id"] == slide["file_id"]].copy()
        if slide_patches.empty:
            continue
        base_path = Path(slide[args.base_column])
        out_path = base_path.parent / args.image_name
        risk_overlay_path = overlay_for_slide(slide, slide_patches, out_path, args.base_column, args.top_n)
        if risk_overlay_path:
            rows.append(
                {
                    "case_submitter_id": slide["case_submitter_id"],
                    "project_id": slide["project_id"],
                    "file_id": slide["file_id"],
                    "risk_overlay_path": risk_overlay_path,
                    "patches_drawn": min(args.top_n, len(slide_patches)),
                }
            )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_summary, sep="\t", index=False)
    print(f"wrote {len(rows)} WSI risk overlays to {args.out_summary}")


if __name__ == "__main__":
    main()
