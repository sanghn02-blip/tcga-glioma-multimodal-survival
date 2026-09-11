#!/usr/bin/env python3
import os
from pathlib import Path


def load_dotenv(path=Path(".env")):
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def mysql_config(database=True):
    load_dotenv()
    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3307")),
        "user": os.getenv("MYSQL_USER", "tcga"),
        "password": os.getenv("MYSQL_PASSWORD", "tcga_password"),
        "charset": "utf8mb4",
        "autocommit": False,
        "local_infile": False,
    }
    if database:
        config["database"] = os.getenv("MYSQL_DATABASE", "tcga_glioma")
    return config
