#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pymysql

from db_config import load_dotenv, mysql_config


def split_sql(sql):
    statements = []
    current = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip(";"))
            current = []
    if current:
        statements.append("\n".join(current))
    return statements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, default=Path("sql/schema.sql"))
    args = parser.parse_args()

    load_dotenv()
    config = mysql_config(database=False)
    database = os.getenv("MYSQL_DATABASE", "tcga_glioma")
    connection = pymysql.connect(**config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE `{database}`")
            for statement in split_sql(args.schema.read_text()):
                cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()
    print(f"initialized MySQL schema in database `{database}`")


if __name__ == "__main__":
    main()
