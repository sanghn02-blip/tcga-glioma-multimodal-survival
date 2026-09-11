#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from run_cox_baseline import load_cases, load_wsi_features, select_wsi_features


MEAN_FEATURE_RE = re.compile(r"^emb_(\d{4})_mean$")


def fit_wsi_cox(cases, wsi, n_features, penalizer, l1_ratio):
    selected = select_wsi_features(wsi, n_features)
    selected = [feature for feature in selected if feature in wsi.columns]
    if not selected:
        raise ValueError("No WSI embedding features were selected.")

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(wsi[selected]), index=wsi.index, columns=selected)
    train_frame = pd.concat([cases[["os_days", "os_event"]], x], axis=1)
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    model.fit(train_frame, duration_col="os_days", event_col="os_event")
    return model, scaler, selected


def selected_mean_dimensions(selected_features):
    dims = []
    for feature in selected_features:
        match = MEAN_FEATURE_RE.match(feature)
        if match:
            dims.append((feature, int(match.group(1))))
    return dims


def score_patches(patch_meta, patch_matrix, model, scaler, selected_features):
    mean_dims = selected_mean_dimensions(selected_features)
    if not mean_dims:
        raise ValueError("Selected WSI features did not include emb_####_mean columns.")

    coef = model.params_.reindex(selected_features).fillna(0.0)
    mean_by_feature = dict(zip(selected_features, scaler.mean_))
    scale_by_feature = dict(zip(selected_features, scaler.scale_))

    score = np.zeros(len(patch_meta), dtype=float)
    abs_score = np.zeros(len(patch_meta), dtype=float)
    contribution_rows = []
    for feature, dim in mean_dims:
        scale = scale_by_feature.get(feature, 1.0) or 1.0
        values = (patch_matrix[:, dim].astype(float) - mean_by_feature.get(feature, 0.0)) / scale
        contrib = values * float(coef.get(feature, 0.0))
        score += contrib
        abs_score += np.abs(contrib)
        contribution_rows.append(
            {
                "feature_name": feature,
                "embedding_dim": dim,
                "coef": float(coef.get(feature, 0.0)),
                "scaler_mean": float(mean_by_feature.get(feature, 0.0)),
                "scaler_scale": float(scale),
            }
        )

    out = patch_meta.copy()
    out["patch_risk_score"] = score
    out["patch_importance_abs"] = abs_score
    out["patch_risk_direction"] = np.where(out["patch_risk_score"] >= 0, "raises_risk", "lowers_risk")
    out["patch_risk_rank"] = (
        out.groupby("file_id")["patch_risk_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["patch_importance_rank"] = (
        out.groupby("file_id")["patch_importance_abs"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    max_abs = out.groupby("file_id")["patch_importance_abs"].transform("max").replace(0, np.nan)
    out["patch_importance_norm"] = (out["patch_importance_abs"] / max_abs).fillna(0.0)
    return out.sort_values(["file_id", "patch_risk_rank"]), pd.DataFrame(contribution_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("outputs/tcga_lgg_gbm_wsi_dev_50_cases.tsv"))
    parser.add_argument("--slide-embeddings", type=Path, default=Path("data/processed/wsi_dev_50/embeddings/uni_MahmoodLab_UNI_slide_embeddings.tsv"))
    parser.add_argument("--patch-embeddings", type=Path, default=Path("data/processed/wsi_dev_50/embeddings/uni_MahmoodLab_UNI_patch_embeddings.npz"))
    parser.add_argument("--patch-metadata", type=Path, default=Path("data/processed/wsi_dev_50/embeddings/uni_MahmoodLab_UNI_patch_metadata.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/wsi_uni_patch_importance.tsv"))
    parser.add_argument("--coef-out", type=Path, default=Path("results/wsi_uni_patch_importance_coefficients.tsv"))
    parser.add_argument("--n-wsi-features", type=int, default=50)
    parser.add_argument("--penalizer", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    wsi = load_wsi_features(args.slide_embeddings, cases.index)
    common = cases.index.intersection(wsi.dropna(how="all").index)
    cases = cases.loc[common].copy()
    wsi = wsi.reindex(common).fillna(0)

    model, scaler, selected = fit_wsi_cox(cases, wsi, args.n_wsi_features, args.penalizer, args.l1_ratio)
    patch_meta = pd.read_csv(args.patch_metadata, sep="\t")
    patch_matrix = np.load(args.patch_embeddings)["embeddings"]
    if len(patch_meta) != patch_matrix.shape[0]:
        raise ValueError(f"Patch metadata rows ({len(patch_meta)}) do not match embeddings ({patch_matrix.shape[0]}).")

    scored, coefficients = score_patches(patch_meta, patch_matrix, model, scaler, selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, sep="\t", index=False)
    coefficients.to_csv(args.coef_out, sep="\t", index=False)
    print(
        f"wrote {len(scored)} patch scores across {scored['file_id'].nunique()} slides "
        f"using {len(selected)} WSI features ({len(coefficients)} mean embedding dimensions)"
    )


if __name__ == "__main__":
    main()
