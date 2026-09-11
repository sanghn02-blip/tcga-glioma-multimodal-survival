#!/usr/bin/env python3
import argparse
import csv
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(file_id, filename, expected_md5, out_dir, force=False, timeout=300):
    out_path = out_dir / file_id / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        if expected_md5 and md5sum(out_path) != expected_md5:
            print(f"checksum mismatch, re-downloading: {out_path}")
        else:
            print(f"exists: {out_path}")
            return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(GDC_DATA_URL.format(file_id=file_id), stream=True, timeout=(30, timeout)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with tmp_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=filename[:40],
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    tmp_path.replace(out_path)

    if expected_md5 and md5sum(out_path) != expected_md5:
        raise RuntimeError(f"MD5 checksum failed for {out_path}")
    return out_path


def download_with_retries(row, out_dir, force=False, timeout=300, retries=5, retry_sleep=10):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return download_file(
                file_id=row["id"],
                filename=row["filename"],
                expected_md5=row.get("md5", ""),
                out_dir=out_dir,
                force=force,
                timeout=timeout,
            )
        except (requests.RequestException, RuntimeError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"download failed on attempt {attempt}/{retries}: {row['filename']} ({exc})")
            print(f"retrying in {retry_sleep}s...")
            time.sleep(retry_sleep)
    raise RuntimeError(f"failed after {retries} attempts: {row['filename']}") from last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=300, help="Read timeout in seconds per chunk.")
    parser.add_argument("--retries", type=int, default=5, help="Retry attempts per file.")
    parser.add_argument("--retry-sleep", type=int, default=10, help="Seconds to wait between retries.")
    parser.add_argument("--workers", type=int, default=1, help="Number of files to download in parallel.")
    args = parser.parse_args()

    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if args.workers <= 1:
        for index, row in enumerate(rows, 1):
            print(f"[{index}/{len(rows)}] {row['filename']}")
            download_with_retries(
                row=row,
                out_dir=args.out_dir,
                force=args.force,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
        return

    print(f"downloading {len(rows)} files with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_with_retries,
                row=row,
                out_dir=args.out_dir,
                force=args.force,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            ): row
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                path = future.result()
                print(f"[{index}/{len(rows)}] done: {path}")
            except Exception as exc:
                print(f"[{index}/{len(rows)}] failed: {row['filename']} ({exc})")
                raise


if __name__ == "__main__":
    main()
