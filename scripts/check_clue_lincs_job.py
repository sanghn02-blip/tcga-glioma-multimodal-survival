#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


JOB_STATUS_URL = "https://api.clue.io/api/jobs/findByJobId/{job_id}"


def clean_status(payload):
    if not isinstance(payload, dict):
        return {"raw": payload}
    out = {key: payload.get(key) for key in ["status", "job_id", "download_status", "download_url"] if key in payload}
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ["status", "job_id", "download_status", "download_url", "id"]:
            if key in result and key not in out:
                out[key] = result[key]
    return out


def download_url(value):
    if not value:
        return None
    value = str(value)
    if value.startswith("//"):
        return "https:" + value
    if value.startswith("/"):
        return urljoin("https://api.clue.io", value)
    return value


def fetch_status(job_id, api_key, timeout):
    response = requests.get(
        JOB_STATUS_URL.format(job_id=job_id),
        headers={"user_key": api_key, "Accept": "application/json"},
        timeout=timeout,
    )
    status = {"status_code": response.status_code}
    try:
        status.update(clean_status(response.json()))
    except ValueError:
        status["response_text"] = response.text[:2000]
    response.raise_for_status()
    return status


def download_results(status, out_dir, timeout):
    url = download_url(status.get("download_url"))
    if not url:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "clue_lincs_results.tar.gz"
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    extract_dir = out_dir / "clue_lincs_results"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")
    return {"archive": str(archive_path), "extract_dir": str(extract_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--clue-api-key-env", default="CLUE_API_KEY")
    parser.add_argument("--out-dir", type=Path, default=Path("results/clue_lincs"))
    parser.add_argument("--out-status", type=Path, default=Path("results/clue_lincs/clue_job_status.json"))
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=360)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get(args.clue_api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {args.clue_api_key_env}. Set it before checking the CLUE job.")

    deadline = time.time() + args.max_wait_seconds
    status = None
    while True:
        status = fetch_status(args.job_id, api_key, args.timeout)
        args.out_status.parent.mkdir(parents=True, exist_ok=True)
        args.out_status.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if not args.poll or status.get("download_status") == "completed" or time.time() >= deadline:
            break
        time.sleep(args.poll_seconds)

    if args.download and status.get("download_status") == "completed":
        downloaded = download_results(status, args.out_dir, args.timeout)
        if downloaded:
            status["downloaded"] = downloaded
            args.out_status.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
