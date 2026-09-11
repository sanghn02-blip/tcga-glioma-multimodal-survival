#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


UNI_MODEL = "hf_hub:MahmoodLab/UNI"
UNI2_MODEL = "hf-hub:MahmoodLab/UNI2-h"


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


def load_timm_encoder(model_name, pretrained, device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    kwargs = {}
    if model_name == UNI_MODEL:
        kwargs.update({"init_values": 1.0, "dynamic_img_size": True})
    elif model_name == UNI2_MODEL:
        kwargs.update({"init_values": 1e-5, "dynamic_img_size": True})
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gatedrepoerror" in type(exc).__name__.lower() or "401" in message or "gated repo" in message:
            raise SystemExit(
                "Cannot access the gated Hugging Face pathology encoder.\n"
                "Confirm that the model access request was approved, then authenticate with:\n"
                "  HF_HOME=models/huggingface_cache .venv/bin/hf auth login\n"
                "Then rerun the UNI/UNI2 embedding task."
            ) from exc
        raise
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def load_conch_encoder(checkpoint, token, device):
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "CONCH is not installed. Run:\n"
            "  .venv/bin/python -m pip install -r requirements-conch.txt"
        ) from exc
    try:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint,
            hf_auth_token=token,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gatedrepoerror" in type(exc).__name__.lower() or "401" in message or "403" in message or "gated repo" in message:
            raise SystemExit(
                "Cannot access the gated CONCH Hugging Face repository.\n"
                "Visit https://huggingface.co/MahmoodLab/CONCH, request access, then rerun the CONCH embedding task."
            ) from exc
        raise
    model.eval().to(device)
    return model, preprocess


def load_encoder(args, device):
    torch = require_torch()
    if args.encoder == "uni":
        return "uni", load_timm_encoder(UNI_MODEL, True, device), torch
    if args.encoder == "uni2":
        return "uni2", load_timm_encoder(UNI2_MODEL, True, device), torch
    if args.encoder == "conch":
        token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        checkpoint = args.checkpoint or "hf_hub:MahmoodLab/conch"
        return "conch", load_conch_encoder(checkpoint, token, device), torch
    pretrained = not args.no_pretrained
    return "timm", load_timm_encoder(args.model_name, pretrained, device), torch


def encode_batch(model, images, encoder_name, torch, device):
    batch = torch.stack(images).to(device)
    with torch.inference_mode():
        if encoder_name == "conch":
            values = model.encode_image(batch, proj_contrast=False, normalize=False)
        else:
            values = model(batch)
    if isinstance(values, (tuple, list)):
        values = values[0]
    values = values.detach().float().cpu().numpy()
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    return values


def aggregate_slide_embeddings(patches, vectors):
    patch_frame = patches[["case_submitter_id", "project_id", "file_id", "file_name"]].reset_index(drop=True)
    rows = []
    for file_id, indexes in patch_frame.groupby("file_id").groups.items():
        group = patch_frame.loc[list(indexes)]
        arr = vectors[list(indexes)]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        rows.append(
            {
                "case_submitter_id": group.iloc[0]["case_submitter_id"],
                "project_id": group.iloc[0]["project_id"],
                "file_id": file_id,
                "file_name": group.iloc[0]["file_name"],
                "patches_embedded": int(arr.shape[0]),
                **{f"emb_{i:04d}_mean": float(value) for i, value in enumerate(mean)},
                **{f"emb_{i:04d}_std": float(value) for i, value in enumerate(std)},
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-image-index", type=Path, default=Path("data/processed/wsi_dev_50/patch_images/patch_image_index.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wsi_dev_50/embeddings"))
    parser.add_argument("--encoder", choices=["timm", "uni", "uni2", "conch"], default="timm")
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--checkpoint", help="CONCH checkpoint path or hf_hub: repo id.")
    parser.add_argument("--hf-token", help="Hugging Face token. Prefer HF_TOKEN env var.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-patches", type=int, help="Optional smoke-test limit.")
    parser.add_argument("--no-pretrained", action="store_true", help="Only for --encoder timm.")
    args = parser.parse_args()

    torch = require_torch()
    device = resolve_device(torch, args.device)
    encoder_name, (model, transform), torch = load_encoder(args, device)
    patches = pd.read_csv(args.patch_image_index, sep="\t")
    if args.max_patches:
        patches = patches.head(args.max_patches).copy()
    if patches.empty:
        raise ValueError("No patch images found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    processed_rows = []
    batch = []
    batch_rows = []
    for _, row in patches.iterrows():
        image_path = Path(row["patch_image_path"])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        batch.append(transform(image))
        batch_rows.append(row)
        if len(batch) >= args.batch_size:
            vectors.append(encode_batch(model, batch, encoder_name, torch, device))
            processed_rows.extend(batch_rows)
            batch = []
            batch_rows = []
    if batch:
        vectors.append(encode_batch(model, batch, encoder_name, torch, device))
        processed_rows.extend(batch_rows)
    if not vectors:
        raise ValueError("No patch images could be embedded.")

    matrix = np.concatenate(vectors, axis=0)
    embedded = pd.DataFrame(processed_rows).reset_index(drop=True)
    patch_meta = embedded.drop(columns=[], errors="ignore").copy()
    patch_meta["embedding_row"] = np.arange(len(patch_meta))
    slide_embeddings = aggregate_slide_embeddings(embedded, matrix)

    model_id = {
        "uni": "MahmoodLab_UNI",
        "uni2": "MahmoodLab_UNI2-h",
        "conch": "MahmoodLab_CONCH",
    }.get(args.encoder, args.model_name)
    prefix = f"{args.encoder}_{model_id.replace('/', '_').replace(':', '_')}"
    np.savez_compressed(args.out_dir / f"{prefix}_patch_embeddings.npz", embeddings=matrix)
    patch_meta.to_csv(args.out_dir / f"{prefix}_patch_metadata.tsv", sep="\t", index=False)
    slide_embeddings.to_csv(args.out_dir / f"{prefix}_slide_embeddings.tsv", sep="\t", index=False)
    summary = {
        "encoder": args.encoder,
        "model_name": model_id,
        "device": device,
        "patches_embedded": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "slides_embedded": int(len(slide_embeddings)),
        "slide_embeddings": str(args.out_dir / f"{prefix}_slide_embeddings.tsv"),
        "patch_embeddings": str(args.out_dir / f"{prefix}_patch_embeddings.npz"),
        "patch_metadata": str(args.out_dir / f"{prefix}_patch_metadata.tsv"),
    }
    (args.out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
