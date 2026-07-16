#!/usr/bin/env python3
"""Merge JSONL(.gz) shards without loading them into memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = write_jsonl(args.output, (row for path in args.inputs for row in read_jsonl(path)))
    print(f"wrote {count:,} rows to {args.output}")


if __name__ == "__main__":
    main()
