#!/usr/bin/env python3
import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = "https://api.gdc.cancer.gov"
CBIO_BASE = "https://www.cbioportal.org/api"
PROJECTS = ["TCGA-LGG", "TCGA-GBM"]
CBIO_STUDIES = ["lgg_tcga_pan_can_atlas_2018", "gbm_tcga_pan_can_atlas_2018"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def post(endpoint, payload):
    req = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def fetch(endpoint, filters, fields, size=2000):
    records = []
    offset = 0
    while True:
        payload = {
            "filters": filters,
            "fields": ",".join(fields),
            "format": "JSON",
            "size": size,
            "from": offset,
        }
        data = post(endpoint, payload)["data"]
        records.extend(data["hits"])
        total = data["pagination"]["total"]
        offset += size
        if len(records) >= total:
            return records


def inop(field, values):
    return {"op": "in", "content": {"field": field, "value": values}}


def and_filter(content):
    return {"op": "and", "content": content}


def case_project(records):
    mapping = {}
    for record in records:
        for case in record.get("cases", []):
            mapping[case["submitter_id"]] = case["project"]["project_id"]
    return mapping


def case_id(record):
    return record["cases"][0]["submitter_id"]


def project_id(record):
    return record["cases"][0]["project"]["project_id"]


def sample_submitters(record):
    values = []
    for case in record.get("cases", []):
        for sample in case.get("samples", []):
            values.append(sample.get("submitter_id", ""))
    return ";".join(v for v in values if v)


def os_info(case):
    demographic = case.get("demographic") or {}
    vital_status = (demographic.get("vital_status") or "").lower()
    if vital_status == "dead":
        event = 1
    elif vital_status == "alive":
        event = 0
    else:
        event = None

    days = demographic.get("days_to_death")
    if days is None:
        followup_days = []
        for diagnosis in case.get("diagnoses") or []:
            if diagnosis.get("days_to_last_follow_up") is not None:
                followup_days.append(diagnosis["days_to_last_follow_up"])
        for follow_up in case.get("follow_ups") or []:
            if follow_up.get("days_to_follow_up") is not None:
                followup_days.append(follow_up["days_to_follow_up"])
        days = max(followup_days) if followup_days else None
    return event, days


def manifest_rows(records, eligible_cases):
    rows = []
    for record in records:
        if case_id(record) not in eligible_cases:
            continue
        rows.append(
            {
                "id": record["file_id"],
                "filename": record["file_name"],
                "md5": record.get("md5sum", ""),
                "size": record.get("file_size", ""),
                "state": record.get("state", ""),
            }
        )
    return rows


def metadata_rows(records, eligible_cases):
    rows = []
    for record in records:
        submitter = case_id(record)
        if submitter not in eligible_cases:
            continue
        rows.append(
            {
                "file_id": record["file_id"],
                "file_name": record["file_name"],
                "case_submitter_id": submitter,
                "project_id": project_id(record),
                "sample_submitter_ids": sample_submitters(record),
                "data_type": record.get("data_type", ""),
                "experimental_strategy": record.get("experimental_strategy", ""),
                "workflow_type": (record.get("analysis") or {}).get("workflow_type", ""),
                "file_size": record.get("file_size", ""),
            }
        )
    return rows


def write_tsv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def cbio_get(path):
    with urllib.request.urlopen(CBIO_BASE + path, timeout=90) as response:
        return json.load(response)


def cbio_covariates():
    rows = {}
    for study_id in CBIO_STUDIES:
        patient_data = cbio_get(
            f"/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000"
        )
        sample_data = cbio_get(
            f"/studies/{study_id}/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY&pageSize=100000"
        )
        for item in patient_data:
            patient_id = item["patientId"]
            row = rows.setdefault(patient_id, {"case_submitter_id": patient_id, "cbioportal_study_id": study_id})
            if item.get("clinicalAttributeId") == "SUBTYPE":
                row["pancan_subtype"] = item.get("value", "")
        for item in sample_data:
            patient_id = item["patientId"]
            row = rows.setdefault(patient_id, {"case_submitter_id": patient_id, "cbioportal_study_id": study_id})
            if item.get("clinicalAttributeId") == "GRADE":
                row["histologic_grade"] = item.get("value", "")
            if item.get("clinicalAttributeId") == "CANCER_TYPE_DETAILED":
                row["cancer_type_detailed"] = item.get("value", "")
            if item.get("clinicalAttributeId") == "TUMOR_TYPE":
                row["tumor_type"] = item.get("value", "")
    for row in rows.values():
        row.setdefault("pancan_subtype", "")
        row.setdefault("histologic_grade", "")
        row.setdefault("cancer_type_detailed", "")
        row.setdefault("tumor_type", "")
    return sorted(rows.values(), key=lambda row: row["case_submitter_id"])


def main():
    status = json.load(urllib.request.urlopen(BASE + "/status", timeout=30))
    common = [
        inop("cases.project.project_id", PROJECTS),
        inop("cases.samples.sample_type", ["Primary Tumor"]),
        inop("access", ["open"]),
    ]
    file_fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "state",
        "data_type",
        "data_format",
        "experimental_strategy",
        "analysis.workflow_type",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
    ]
    wsi = fetch(
        "/files",
        and_filter(
            common
            + [
                inop("data_type", ["Slide Image"]),
                inop("experimental_strategy", ["Diagnostic Slide"]),
            ]
        ),
        file_fields,
    )
    rna = fetch(
        "/files",
        and_filter(
            common
            + [
                inop("data_category", ["Transcriptome Profiling"]),
                inop("data_type", ["Gene Expression Quantification"]),
                inop("analysis.workflow_type", ["STAR - Counts"]),
            ]
        ),
        file_fields,
    )

    case_fields = [
        "submitter_id",
        "project.project_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.gender",
        "diagnoses.age_at_diagnosis",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.tumor_grade",
        "diagnoses.primary_diagnosis",
        "diagnoses.tissue_or_organ_of_origin",
        "follow_ups.days_to_follow_up",
    ]
    cases = fetch(
        "/cases",
        inop("project.project_id", PROJECTS),
        case_fields,
    )
    clinical = {case["submitter_id"]: case for case in cases}

    wsi_projects = case_project(wsi)
    rna_projects = case_project(rna)
    overlap = sorted(set(wsi_projects) & set(rna_projects))
    eligible = []
    for case in overlap:
        event, days = os_info(clinical.get(case, {}))
        if event is not None and days not in (None, 0):
            eligible.append(case)

    wsi_by_case = defaultdict(list)
    rna_by_case = defaultdict(list)
    for record in wsi:
        wsi_by_case[case_id(record)].append(record)
    for record in rna:
        rna_by_case[case_id(record)].append(record)

    patient_rows = []
    for case in eligible:
        clinical_case = clinical.get(case, {})
        diagnosis = (clinical_case.get("diagnoses") or [{}])[0]
        demographic = clinical_case.get("demographic") or {}
        event, days = os_info(clinical_case)
        patient_rows.append(
            {
                "case_submitter_id": case,
                "project_id": wsi_projects[case],
                "os_event": event,
                "os_days": days,
                "vital_status": demographic.get("vital_status", ""),
                "gender": demographic.get("gender", ""),
                "age_at_diagnosis_days": diagnosis.get("age_at_diagnosis", ""),
                "tumor_grade": diagnosis.get("tumor_grade", ""),
                "primary_diagnosis": diagnosis.get("primary_diagnosis", ""),
                "wsi_file_count": len(wsi_by_case[case]),
                "rna_file_count": len(rna_by_case[case]),
            }
        )

    dev_cases = []
    for project in PROJECTS:
        project_cases = [row["case_submitter_id"] for row in patient_rows if row["project_id"] == project]
        dev_cases.extend(project_cases[:10])
    dev_case_set = set(dev_cases)

    wsi_manifest = manifest_rows(wsi, set(eligible))
    rna_manifest = manifest_rows(rna, set(eligible))
    dev_wsi_manifest = manifest_rows(wsi, dev_case_set)
    dev_rna_manifest = manifest_rows(rna, dev_case_set)
    wsi_metadata = metadata_rows(wsi, set(eligible))
    rna_metadata = metadata_rows(rna, set(eligible))
    dev_wsi_metadata = metadata_rows(wsi, dev_case_set)
    dev_rna_metadata = metadata_rows(rna, dev_case_set)

    write_tsv(
        OUT / "tcga_lgg_gbm_primary_overlap_cases.tsv",
        patient_rows,
        [
            "case_submitter_id",
            "project_id",
            "os_event",
            "os_days",
            "vital_status",
            "gender",
            "age_at_diagnosis_days",
            "tumor_grade",
            "primary_diagnosis",
            "wsi_file_count",
            "rna_file_count",
        ],
    )
    write_tsv(OUT / "gdc_manifest_rna_star_counts_primary_overlap.tsv", rna_manifest, ["id", "filename", "md5", "size", "state"])
    write_tsv(OUT / "gdc_manifest_wsi_diagnostic_primary_overlap.tsv", wsi_manifest, ["id", "filename", "md5", "size", "state"])
    metadata_fields = [
        "file_id",
        "file_name",
        "case_submitter_id",
        "project_id",
        "sample_submitter_ids",
        "data_type",
        "experimental_strategy",
        "workflow_type",
        "file_size",
    ]
    write_tsv(OUT / "gdc_rna_file_map_primary_overlap.tsv", rna_metadata, metadata_fields)
    write_tsv(OUT / "gdc_wsi_file_map_primary_overlap.tsv", wsi_metadata, metadata_fields)
    write_tsv(
        OUT / "tcga_lgg_gbm_dev_20_cases.tsv",
        [row for row in patient_rows if row["case_submitter_id"] in dev_case_set],
        [
            "case_submitter_id",
            "project_id",
            "os_event",
            "os_days",
            "vital_status",
            "gender",
            "age_at_diagnosis_days",
            "tumor_grade",
            "primary_diagnosis",
            "wsi_file_count",
            "rna_file_count",
        ],
    )
    write_tsv(OUT / "gdc_manifest_rna_dev_20_cases.tsv", dev_rna_manifest, ["id", "filename", "md5", "size", "state"])
    write_tsv(OUT / "gdc_manifest_wsi_dev_20_cases.tsv", dev_wsi_manifest, ["id", "filename", "md5", "size", "state"])
    write_tsv(OUT / "gdc_rna_file_map_dev_20_cases.tsv", dev_rna_metadata, metadata_fields)
    write_tsv(OUT / "gdc_wsi_file_map_dev_20_cases.tsv", dev_wsi_metadata, metadata_fields)
    cbio_rows = cbio_covariates()
    cbio_by_case = {row["case_submitter_id"]: row for row in cbio_rows}
    cbio_eligible = [cbio_by_case[case] for case in eligible if case in cbio_by_case]
    write_tsv(
        OUT / "cbioportal_pancan_glioma_covariates.tsv",
        cbio_rows,
        [
            "case_submitter_id",
            "cbioportal_study_id",
            "pancan_subtype",
            "histologic_grade",
            "cancer_type_detailed",
            "tumor_type",
        ],
    )

    summary = {
        "gdc_status": status,
        "projects": PROJECTS,
        "filters": {
            "sample_type": "Primary Tumor",
            "access": "open",
            "wsi": "data_type=Slide Image, experimental_strategy=Diagnostic Slide",
            "rna": "data_type=Gene Expression Quantification, workflow_type=STAR - Counts",
        },
        "wsi": {
            "files": len(wsi),
            "unique_cases": len(wsi_projects),
            "cases_by_project": dict(Counter(wsi_projects.values())),
            "size_gb": round(sum(row.get("file_size") or 0 for row in wsi) / 1e9, 2),
        },
        "rna": {
            "files": len(rna),
            "unique_cases": len(rna_projects),
            "cases_by_project": dict(Counter(rna_projects.values())),
            "size_gb": round(sum(row.get("file_size") or 0 for row in rna) / 1e9, 2),
        },
        "overlap_primary_wsi_rna_cases": {
            "total": len(overlap),
            "by_project": dict(Counter(wsi_projects[case] for case in overlap)),
        },
        "eligible_primary_wsi_rna_survival_cases": {
            "total": len(eligible),
            "by_project": dict(Counter(wsi_projects[case] for case in eligible)),
        },
        "eligible_manifests": {
            "wsi_files": len(wsi_manifest),
            "wsi_size_gb": round(sum(int(row["size"]) for row in wsi_manifest if row["size"]) / 1e9, 2),
            "rna_files": len(rna_manifest),
            "rna_size_gb": round(sum(int(row["size"]) for row in rna_manifest if row["size"]) / 1e9, 2),
        },
        "development_subset": {
            "cases": len(dev_cases),
            "cases_by_project": dict(Counter(row["project_id"] for row in patient_rows if row["case_submitter_id"] in dev_case_set)),
            "wsi_files": len(dev_wsi_manifest),
            "wsi_size_gb": round(sum(int(row["size"]) for row in dev_wsi_manifest if row["size"]) / 1e9, 2),
            "rna_files": len(dev_rna_manifest),
            "rna_size_gb": round(sum(int(row["size"]) for row in dev_rna_manifest if row["size"]) / 1e9, 2),
        },
        "cbioportal_covariates": {
            "studies": CBIO_STUDIES,
            "patients": len(cbio_rows),
            "eligible_patient_id_coverage": f"{len(cbio_eligible)}/{len(eligible)}",
            "eligible_subtypes": dict(
                Counter(row["pancan_subtype"] for row in cbio_eligible if row["pancan_subtype"])
            ),
            "eligible_grades": dict(
                Counter(row["histologic_grade"] for row in cbio_eligible if row["histologic_grade"])
            ),
            "subtypes": dict(Counter(row["pancan_subtype"] for row in cbio_rows if row["pancan_subtype"])),
            "grades": dict(Counter(row["histologic_grade"] for row in cbio_rows if row["histologic_grade"])),
        },
    }
    (OUT / "gdc_inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    markdown = f"""# TCGA LGG/GBM WSI + RNA Data Inventory

Checked against GDC `{status['data_release']}`.

## Recommended First Cohort

Use primary tumor cases with all three required pieces:

- Diagnostic whole-slide image from GDC
- RNA-seq gene expression quantification from GDC STAR - Counts
- Overall survival event/time from GDC clinical fields

This gives **{len(eligible)} eligible patients**: {dict(Counter(wsi_projects[case] for case in eligible))}.

## GDC Counts

| Data | Files | Unique cases | Size |
|---|---:|---:|---:|
| Diagnostic WSI, primary tumor | {len(wsi)} | {len(wsi_projects)} | {summary['wsi']['size_gb']} GB |
| RNA STAR - Counts, primary tumor | {len(rna)} | {len(rna_projects)} | {summary['rna']['size_gb']} GB |
| WSI + RNA overlap | - | {len(overlap)} | - |
| WSI + RNA + survival | - | {len(eligible)} | - |

## Generated Files

- `tcga_lgg_gbm_primary_overlap_cases.tsv`: eligible patient list and basic clinical/survival fields.
- `gdc_manifest_rna_star_counts_primary_overlap.tsv`: RNA STAR-counts manifest for eligible patients.
- `gdc_manifest_wsi_diagnostic_primary_overlap.tsv`: diagnostic WSI manifest for eligible patients.
- `gdc_rna_file_map_primary_overlap.tsv`, `gdc_wsi_file_map_primary_overlap.tsv`: file-to-patient metadata for modeling.
- `gdc_inventory_summary.json`: machine-readable summary of this inventory.
- `cbioportal_pancan_glioma_covariates.tsv`: PanCancer Atlas subtype/grade helper table for IDH/1p19q-aware analyses.
- `tcga_lgg_gbm_dev_20_cases.tsv`, `gdc_manifest_*_dev_20_cases.tsv`, `gdc_*_file_map_dev_20_cases.tsv`: small balanced development subset.

## Download Commands

```bash
gdc-client download -m outputs/gdc_manifest_rna_star_counts_primary_overlap.tsv -d data/gdc/rna
gdc-client download -m outputs/gdc_manifest_wsi_diagnostic_primary_overlap.tsv -d data/gdc/wsi
```

The WSI manifest is large: **{summary['eligible_manifests']['wsi_files']} slide files / {summary['eligible_manifests']['wsi_size_gb']} GB**. For development, start with a small subset of slides, validate tissue detection and patching, then scale up.

Development subset: **{summary['development_subset']['cases']} patients**, **{summary['development_subset']['wsi_files']} WSI files / {summary['development_subset']['wsi_size_gb']} GB**, **{summary['development_subset']['rna_files']} RNA files / {summary['development_subset']['rna_size_gb']} GB**.

## Practical Starting Recommendation

1. Start with RNA + clinical survival baselines using `tcga_lgg_gbm_primary_overlap_cases.tsv` and the RNA manifest.
2. Download 20-50 WSI files first for patch extraction and feature extraction tests.
3. Freeze patient splits at the case level before generating patch embeddings.
4. Add WSI-only and WSI+RNA fusion after the RNA/clinical Cox baseline is working.
5. Treat LGG vs GBM confounding explicitly with project/grade/IDH-aware analysis.

## Useful External Clinical Covariates

cBioPortal PanCancer Atlas studies provide convenient derived covariates:

- `lgg_tcga_pan_can_atlas_2018`
- `gbm_tcga_pan_can_atlas_2018`

The generated `cbioportal_pancan_glioma_covariates.tsv` includes `pancan_subtype`, e.g. `LGG_IDHmut-codel`, `LGG_IDHmut-non-codel`, `LGG_IDHwt`, `GBM_IDHwt`.

Among the 681 eligible GDC patients, cBioPortal covariates match {len(cbio_eligible)} patients.
"""
    (OUT / "data_inventory.md").write_text(markdown)


if __name__ == "__main__":
    main()
