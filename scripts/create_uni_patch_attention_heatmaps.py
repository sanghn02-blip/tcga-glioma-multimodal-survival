#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


UNI_MODEL = "hf_hub:MahmoodLab/UNI"


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run:\n"
            "  .venv/bin/python -m pip install -r requirements-encoder.txt"
        ) from exc
    return torch


def resolve_device(torch, requested):
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_uni(device, offline):
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    try:
        model = timm.create_model(
            UNI_MODEL,
            pretrained=True,
            num_classes=0,
            init_values=1.0,
            dynamic_img_size=True,
        )
    except Exception as exc:
        raise SystemExit(
            "Cannot load UNI from Hugging Face cache/access.\n"
            "If this is the first run, authenticate and extract UNI embeddings first:\n"
            "  HF_HOME=models/huggingface_cache .venv/bin/hf auth login\n"
            "  HF_HOME=models/huggingface_cache .venv/bin/python scripts/extract_wsi_patch_embeddings.py --encoder uni\n"
        ) from exc

    model.eval().to(device)
    for block in getattr(model, "blocks", []):
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def install_attention_hooks(model):
    captured = []
    handles = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().float().cpu())

    for block in getattr(model, "blocks", []):
        handles.append(block.attn.attn_drop.register_forward_hook(hook))
    return captured, handles


def attention_rollout(attentions, discard_ratio):
    if not attentions:
        raise ValueError("No attention maps were captured.")
    rollout = None
    for attn in attentions:
        # attn: batch x heads x tokens x tokens
        fused = attn.mean(dim=1)
        if discard_ratio > 0:
            flat = fused.view(fused.shape[0], -1)
            threshold = int(flat.shape[1] * discard_ratio)
            if threshold > 0:
                indexes = flat.argsort(dim=1)[:, :threshold]
                flat.scatter_(1, indexes, 0)
                fused = flat.view_as(fused)
        eye = np.eye(fused.shape[-1], dtype=np.float32)
        fused_np = fused.numpy() + eye[None, :, :]
        fused_np = fused_np / np.maximum(fused_np.sum(axis=-1, keepdims=True), 1e-6)
        rollout = fused_np if rollout is None else np.matmul(fused_np, rollout)
    return rollout


def colorize_heatmap(values, alpha=120):
    values = np.clip(values, 0, 1)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (210 * (1 - values)).astype(np.uint8)
    rgba[..., 2] = (40 * (1 - values)).astype(np.uint8)
    rgba[..., 3] = (alpha * values).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def normalize_map(values):
    values = values.astype(np.float32)
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def draw_border(image, direction):
    draw = ImageDraw.Draw(image)
    color = (220, 62, 48) if direction == "raises_risk" else (38, 99, 235)
    width, height = image.size
    for offset in range(4):
        draw.rectangle((offset, offset, width - 1 - offset, height - 1 - offset), outline=color)
    return image


def heatmap_for_batch(model, transform, rows, torch, device, discard_ratio, alpha):
    captured, handles = install_attention_hooks(model)
    try:
        tensors = []
        originals = []
        for _, row in rows.iterrows():
            image = Image.open(row["patch_image_path"]).convert("RGB")
            originals.append(image)
            tensors.append(transform(image))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            _ = model(batch)
        rollout = attention_rollout(captured, discard_ratio)
    finally:
        for handle in handles:
            handle.remove()

    heatmaps = []
    for idx, image in enumerate(originals):
        cls_attention = rollout[idx, 0, 1:]
        grid = int(math.sqrt(cls_attention.shape[0]))
        if grid * grid != cls_attention.shape[0]:
            side = int(math.sqrt(cls_attention.shape[0]))
            cls_attention = cls_attention[: side * side]
            grid = side
        values = normalize_map(cls_attention.reshape(grid, grid))
        hm = Image.fromarray((values * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BICUBIC)
        hm_values = np.asarray(hm).astype(np.float32) / 255.0
        overlay = Image.alpha_composite(image.convert("RGBA"), colorize_heatmap(hm_values, alpha=alpha)).convert("RGB")
        heatmaps.append(overlay)
    return heatmaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-importance", type=Path, default=Path("results/wsi_uni_patch_importance.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wsi_dev_50/patch_heatmaps_uni"))
    parser.add_argument("--out-index", type=Path, default=Path("results/wsi_uni_patch_heatmaps.tsv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--top-n-per-slide", type=int, default=8)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    parser.add_argument("--alpha", type=int, default=132)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch = require_torch()
    device = resolve_device(torch, args.device)
    model, transform = load_uni(device, args.offline)

    patches = pd.read_csv(args.patch_importance, sep="\t")
    patches = patches.sort_values(["file_id", "patch_risk_rank", "rank"], na_position="last")
    if args.top_n_per_slide:
        patches = patches.groupby("file_id", group_keys=False).head(args.top_n_per_slide)
    if args.max_patches:
        patches = patches.head(args.max_patches).copy()

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(patches), args.batch_size):
        batch_rows = patches.iloc[start : start + args.batch_size]
        overlays = heatmap_for_batch(model, transform, batch_rows, torch, device, args.discard_ratio, args.alpha)
        for (_, row), overlay in zip(batch_rows.iterrows(), overlays):
            out_dir = args.out_dir / str(row["file_id"])
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(row["patch_image_path"]).stem
            out_path = out_dir / f"{stem}_attention_heatmap.jpg"
            overlay = draw_border(overlay, row.get("patch_risk_direction"))
            overlay.save(out_path, quality=92)
            rows.append(
                {
                    "case_submitter_id": row["case_submitter_id"],
                    "project_id": row["project_id"],
                    "file_id": row["file_id"],
                    "rank": row["rank"],
                    "patch_risk_rank": row.get("patch_risk_rank"),
                    "patch_risk_score": row.get("patch_risk_score"),
                    "patch_risk_direction": row.get("patch_risk_direction"),
                    "patch_attention_heatmap_path": str(out_path),
                }
            )
        print(f"[{min(start + args.batch_size, len(patches))}/{len(patches)}] heatmaps written")

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_index, sep="\t", index=False)
    print(
        json.dumps(
            {
                "device": device,
                "patch_heatmaps": len(rows),
                "slides": int(pd.DataFrame(rows)["file_id"].nunique()) if rows else 0,
                "out_index": str(args.out_index),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
