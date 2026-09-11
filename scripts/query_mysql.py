#!/usr/bin/env python3
import argparse

import pymysql

from db_config import mysql_config


QUERIES = {
    "summary": [
        ("patients", "SELECT COUNT(*) AS n FROM patients"),
        ("patients_by_project", "SELECT project_id, COUNT(*) AS n FROM patients GROUP BY project_id ORDER BY project_id"),
        ("files_by_type", "SELECT data_type, COUNT(*) AS n, ROUND(SUM(file_size)/1000000000, 2) AS size_gb FROM gdc_files GROUP BY data_type"),
        ("subtypes", "SELECT pancan_subtype, COUNT(*) AS n FROM cbioportal_covariates WHERE pancan_subtype IS NOT NULL GROUP BY pancan_subtype ORDER BY n DESC"),
        ("model_metrics", "SELECT run_id, metric_name, metric_value, metric_text FROM model_metrics ORDER BY run_id, metric_name"),
    ],
    "cohort": [
        ("eligible_cases", "SELECT p.case_submitter_id, p.project_id, p.os_event, p.os_days, c.pancan_subtype, c.histologic_grade FROM patients p LEFT JOIN cbioportal_covariates c USING (case_submitter_id) ORDER BY p.project_id, p.case_submitter_id LIMIT 50"),
    ],
    "predictions": [
        ("dev_predictions", "SELECT case_submitter_id, project_id, risk_score, risk_group, os_days, os_event FROM model_predictions WHERE run_id='rna_dev_cox_v1' ORDER BY risk_score DESC"),
    ],
}


def print_rows(cursor, title, query):
    print(f"\n## {title}")
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    print("\t".join(columns))
    for row in cursor.fetchall():
        print("\t".join("" if value is None else str(value) for value in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", choices=sorted(QUERIES), nargs="?", default="summary")
    args = parser.parse_args()

    connection = pymysql.connect(**mysql_config())
    try:
        with connection.cursor() as cursor:
            for title, query in QUERIES[args.query]:
                print_rows(cursor, title, query)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
