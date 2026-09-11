#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def yes(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def direction_label(direction):
    if direction == "up_in_gbm":
        return "GBM에서 증가"
    if direction == "down_in_gbm":
        return "GBM에서 감소"
    return direction or "방향 미확인"


def mechanism_label(reason):
    labels = {
        "target_inhibits_gbm_up_gene": "GBM에서 증가한 타깃을 억제하는 방향",
        "target_activates_gbm_down_gene": "GBM에서 감소한 타깃을 활성화하는 방향",
        "directional_mechanism_known_but_not_reversal": "작용 방향은 있으나 발현 반전 근거는 약함",
        "mechanism_ambiguous": "약물-타깃 작용 방향 확인 필요",
    }
    return labels.get(str(reason), str(reason or "확인 필요"))


def shortlist_label(row):
    approved = yes(row.get("approved"))
    antineoplastic = yes(row.get("antineoplastic"))
    prior = float(row.get("local_reversal_prior") or 0)
    if approved and antineoplastic and prior >= 1:
        return "우선 검토"
    if approved and prior >= 1:
        return "재창출 검토"
    if prior >= 1:
        return "기전 검토"
    return "보조 후보"


def build_rationale(row):
    target = row.get("primary_target") or "타깃"
    direction = direction_label(row.get("primary_target_direction"))
    log2fc = row.get("primary_target_log2_fc")
    fdr = row.get("primary_target_fdr")
    mechanism = mechanism_label(row.get("local_reversal_reason"))
    pieces = [f"{target}는 {direction}된 유전자이며, 약물 작용은 '{mechanism}'으로 분류됨."]
    try:
        pieces.append(f"발현 차이 log2FC={float(log2fc):.2f}, FDR={float(fdr):.2e}.")
    except (TypeError, ValueError):
        pass
    if yes(row.get("approved")):
        pieces.append("기존 승인 약물이므로 재창출 후보 설명이 쉬움.")
    if yes(row.get("antineoplastic")):
        pieces.append("항암제 이력이 있어 종양 맥락의 선행 근거가 있음.")
    try:
        tau = float(row.get("clue_tau_score"))
        if tau < 0:
            pieces.append(f"CLUE/LINCS tau={tau:.2f}로 GBM 발현 signature를 반전시키는 방향이 확인됨.")
        elif tau > 0:
            pieces.append(f"CLUE/LINCS tau={tau:.2f}로 GBM 발현 signature와 유사한 방향이라 주의가 필요함.")
    except (TypeError, ValueError):
        pass
    return " ".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    default_candidates = Path("results/drug_repurposing_candidates_clue_ready.tsv")
    parser.add_argument("--candidates", type=Path, default=default_candidates)
    parser.add_argument("--deg", type=Path, default=Path("results/gbm_vs_gtex_brain_deg_targeted.tsv"))
    parser.add_argument("--out-shortlist", type=Path, default=Path("results/drug_repurposing_final_shortlist.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/drug_repurposing_final_summary.json"))
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    with_tau = Path("results/drug_repurposing_candidates_with_clue_tau.tsv")
    if args.candidates == default_candidates and with_tau.exists():
        args.candidates = with_tau
    candidates = pd.read_csv(args.candidates, sep="\t")
    deg = pd.read_csv(args.deg, sep="\t")
    sort_col = "final_repurposing_score" if "final_repurposing_score" in candidates.columns else "repurposing_score"
    candidates = candidates.sort_values(sort_col, ascending=False).head(args.top_n).copy()
    candidates["shortlist_rank"] = range(1, len(candidates) + 1)
    candidates["candidate_tier"] = candidates.apply(shortlist_label, axis=1)
    candidates["mechanism_summary_ko"] = candidates["local_reversal_reason"].apply(mechanism_label)
    candidates["rationale_ko"] = candidates.apply(build_rationale, axis=1)
    candidates["validation_status_ko"] = candidates["clue_status"].apply(
        lambda value: "CLUE/LINCS 완료"
        if str(value) == "completed"
        else ("CLUE/LINCS API key 필요" if str(value) == "pending_api_key" else str(value or "확인 필요"))
    )
    candidates["next_validation_ko"] = (
        "CLUE/LINCS tau score로 발현 반전 여부 확인 후, 문헌/BBB 투과성/독성 정보를 붙여 최종 우선순위를 조정"
    )

    keep_cols = [
        "shortlist_rank",
        "candidate_tier",
        "drug_name",
        "primary_target",
        "final_repurposing_score",
        "repurposing_score",
        "primary_target_direction",
        "primary_target_log2_fc",
        "primary_target_fdr",
        "local_reversal_prior",
        "mechanism_summary_ko",
        "approved",
        "antineoplastic",
        "clue_status",
        "clue_tau_score",
        "clue_connectivity_direction",
        "pert_id",
        "clue_pert_iname",
        "validation_status_ko",
        "rationale_ko",
        "next_validation_ko",
    ]
    shortlist = candidates[[col for col in keep_cols if col in candidates.columns]]
    args.out_shortlist.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.out_shortlist, sep="\t", index=False)

    up_genes = int((deg["direction"] == "up_in_gbm").sum()) if "direction" in deg else 0
    down_genes = int((deg["direction"] == "down_in_gbm").sum()) if "direction" in deg else 0
    summary = {
        "comparison": "TCGA-GBM primary tumor vs GTEx normal brain",
        "shortlist_candidates": int(len(shortlist)),
        "deg_genes_tested": int(len(deg)),
        "up_in_gbm_genes": up_genes,
        "down_in_gbm_genes": down_genes,
        "top_candidate": shortlist.iloc[0]["drug_name"] if not shortlist.empty else None,
        "top_candidate_target": shortlist.iloc[0]["primary_target"] if not shortlist.empty else None,
        "top_candidate_tier": shortlist.iloc[0]["candidate_tier"] if not shortlist.empty else None,
        "top_candidate_clue_tau": float(shortlist.iloc[0]["clue_tau_score"]) if not shortlist.empty and "clue_tau_score" in shortlist.columns and pd.notna(shortlist.iloc[0]["clue_tau_score"]) else None,
        "input_candidates": str(args.candidates),
        "output_shortlist": str(args.out_shortlist),
        "note": "Research-only shortlist. CLUE/LINCS tau scores, blood-brain barrier evidence, toxicity, and literature review are required before biological or clinical claims.",
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
