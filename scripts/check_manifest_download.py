#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    complete = 0
    missing = []
    partial = []
    for row in rows:
        expected = args.out_dir / row["id"] / row["filename"]
        part = expected.with_suffix(expected.suffix + ".part")
        if expected.exists():
            complete += 1
        else:
            missing.append(row["filename"])
        if part.exists():
            partial.append(str(part))

    print(f"manifest files: {len(rows)}")
    print(f"complete files: {complete}")
    print(f"missing files: {len(missing)}")
    print(f"partial files: {len(partial)}")
    if partial:
        print("\npartial files:")
        for item in partial[:20]:
            print(item)
    if missing:
        print("\nfirst missing files:")
        for item in missing[:20]:
            print(item)


if __name__ == "__main__":
    main()
