#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,CanonicalSMILES/JSON"
)
CLINICAL_TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_json(url, params=None, timeout=30):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_pubchem_properties(drug_name, timeout=30):
    url = PUBCHEM_PROPERTY_URL.format(name=quote(str(drug_name)))
    payload = get_json(url, timeout=timeout)
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else {}


def fetch_gbm_trials(drug_name, timeout=30):
    studies_by_nct = {}
    for condition in ["glioblastoma", "glioblastoma multiforme", "GBM"]:
        params = {
            "query.cond": condition,
            "query.intr": str(drug_name),
            "pageSize": 20,
            "format": "json",
        }
        payload = get_json(CLINICAL_TRIALS_URL, params=params, timeout=timeout)
        for study in payload.get("studies", []):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            nct_id = ident.get("nctId")
            if not nct_id:
                continue
            status = protocol.get("statusModule", {}).get("overallStatus", "")
            design = protocol.get("designModule", {})
            phases = ";".join(design.get("phases", []) or [])
            brief_title = ident.get("briefTitle", "")
            studies_by_nct[nct_id] = {
                "nct_id": nct_id,
                "status": status,
                "phases": phases,
                "title": brief_title,
            }
    trials = list(studies_by_nct.values())
    trials.sort(key=lambda row: row["nct_id"])
    return trials


def as_float(value):
    try:
        if value is None or str(value).strip() == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def bbb_rule_score(row):
    mw = as_float(row.get("MolecularWeight"))
    xlogp = as_float(row.get("XLogP"))
    tpsa = as_float(row.get("TPSA"))
    hbd = as_float(row.get("HBondDonorCount"))
    hba = as_float(row.get("HBondAcceptorCount"))
    checks = []
    if not np.isnan(mw):
        checks.append(mw <= 450)
    if not np.isnan(xlogp):
        checks.append(1 <= xlogp <= 5)
    if not np.isnan(tpsa):
        checks.append(tpsa <= 90)
    if not np.isnan(hbd):
        checks.append(hbd <= 3)
    if not np.isnan(hba):
        checks.append(hba <= 8)
    if not checks:
        return np.nan, "물성 확인 필요"
    score = sum(1 for value in checks if value) / len(checks)
    if score >= 0.8:
        return score, "BBB 물성 우호적"
    if score >= 0.6:
        return score, "BBB 물성 경계"
    return score, "BBB 물성 불리"


def validation_tier(row):
    score = as_float(row.get("validation_score"))
    bbb = as_float(row.get("bbb_rule_score"))
    trials = int(row.get("gbm_trial_count") or 0)
    if score >= 0.75 and bbb >= 0.6:
        return "우선 검증"
    if trials > 0 or score >= 0.60:
        return "문헌 검토"
    return "보조 후보"


def validation_comment(row):
    parts = [
        f"{row['drug_name']}는 {row.get('primary_target', '')} 기반 후보이며 최종 점수 {row['final_repurposing_score']:.3f}.",
        f"{row.get('bbb_rule_label', '물성 확인 필요')}.",
    ]
    if int(row.get("gbm_trial_count") or 0) > 0:
        parts.append(f"ClinicalTrials.gov에서 GBM 관련 trial {int(row['gbm_trial_count'])}건이 확인됨.")
    else:
        parts.append("ClinicalTrials.gov 기준 직접적인 GBM trial은 확인되지 않음.")
    try:
        tau = float(row.get("clue_tau_score"))
        if tau < 0:
            parts.append(f"CLUE/LINCS tau={tau:.2f}로 발현 반전성이 확인됨.")
        elif tau > 0:
            parts.append(f"CLUE/LINCS tau={tau:.2f}로 발현 유사성이 관찰되어 우선순위 해석에 주의가 필요함.")
    except (TypeError, ValueError):
        parts.append("CLUE/LINCS tau score와 문헌 검토가 다음 검증 단계.")
    return " ".join(parts)


def build_matrix(shortlist, fetch_external, sleep_seconds, timeout):
    rows = []
    for _, row in shortlist.iterrows():
        drug_name = row["drug_name"]
        pubchem = {}
        trials = []
        external_status = "not_fetched"
        if fetch_external:
            try:
                pubchem = fetch_pubchem_properties(drug_name, timeout=timeout)
                trials = fetch_gbm_trials(drug_name, timeout=timeout)
                external_status = "fetched"
            except requests.RequestException as exc:
                external_status = f"fetch_failed: {exc.__class__.__name__}"
            time.sleep(sleep_seconds)
        bbb_score, bbb_label = bbb_rule_score(pubchem)
        clinical_score = min(1.0, len(trials) / 3.0)
        base_score = as_float(row.get("final_repurposing_score"))
        reversal = as_float(row.get("local_reversal_prior"))
        approved_score = 1.0 if truthy(row.get("approved")) else 0.0
        clue_reversal_score = max(0.0, -np.nan_to_num(as_float(row.get("clue_tau_score"))) / 100.0)
        validation_score = (
            0.35 * np.nan_to_num(base_score)
            + 0.15 * np.nan_to_num(reversal)
            + 0.15 * np.nan_to_num(bbb_score)
            + 0.10 * approved_score
            + 0.10 * clinical_score
            + 0.15 * clue_reversal_score
        )
        out = row.to_dict()
        out.update(
            {
                "pubchem_cid": pubchem.get("CID"),
                "molecular_weight": as_float(pubchem.get("MolecularWeight")),
                "xlogp": as_float(pubchem.get("XLogP")),
                "tpsa": as_float(pubchem.get("TPSA")),
                "hbond_donor_count": as_float(pubchem.get("HBondDonorCount")),
                "hbond_acceptor_count": as_float(pubchem.get("HBondAcceptorCount")),
                "canonical_smiles": pubchem.get("CanonicalSMILES") or pubchem.get("ConnectivitySMILES"),
                "bbb_rule_score": bbb_score,
                "bbb_rule_label": bbb_label,
                "gbm_trial_count": len(trials),
                "gbm_trial_nct_ids": ";".join(trial["nct_id"] for trial in trials[:8]),
                "gbm_trial_statuses": ";".join(trial["status"] for trial in trials[:8]),
                "gbm_trial_phases": ";".join(trial["phases"] for trial in trials[:8]),
                "external_lookup_status": external_status,
                "validation_score": validation_score,
            }
        )
        out["validation_tier"] = validation_tier(out)
        out["validation_comment_ko"] = validation_comment(out)
        rows.append(out)
    matrix = pd.DataFrame(rows)
    matrix = matrix.sort_values("validation_score", ascending=False).reset_index(drop=True)
    matrix["validation_rank"] = np.arange(1, len(matrix) + 1)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, default=Path("results/drug_repurposing_final_shortlist.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/drug_repurposing_validation_matrix.tsv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/drug_repurposing_validation_summary.json"))
    parser.add_argument("--fetch-external", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    shortlist = pd.read_csv(args.shortlist, sep="\t")
    matrix = build_matrix(shortlist, args.fetch_external, args.sleep_seconds, args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out, sep="\t", index=False)

    summary = {
        "input_shortlist": str(args.shortlist),
        "validation_candidates": int(len(matrix)),
        "external_fetch": bool(args.fetch_external),
        "pubchem_fetched": int((matrix["external_lookup_status"] == "fetched").sum()),
        "candidates_with_gbm_trials": int((matrix["gbm_trial_count"] > 0).sum()),
        "top_validation_candidate": matrix.iloc[0]["drug_name"] if not matrix.empty else None,
        "top_validation_score": float(matrix.iloc[0]["validation_score"]) if not matrix.empty else None,
        "output": str(args.out),
        "note": "BBB fields are rule-based physicochemical prioritization, not measured brain penetration. ClinicalTrials.gov counts are only a first-pass evidence flag.",
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
