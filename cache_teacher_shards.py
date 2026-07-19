#!/usr/bin/env python3
"""Restartable teacher-cache driver: completed atomic shards are never regenerated."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from runtime import DEVICE_CHOICES
from utils import ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index-corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="teacher")
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cuda")
    parser.add_argument("--model-id")
    parser.add_argument("--mode", choices=("en", "te", "xlit"))
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--score-micro-batch", type=int, default=16)
    parser.add_argument("--index-max-rows", type=int)
    parser.add_argument("--mock-teacher", action="store_true")
    args = parser.parse_args()
    if args.total_examples < 1 or args.shards < 1:
        raise ValueError("total examples and shards must be positive")
    if args.total_examples < args.shards:
        raise ValueError("total examples must be at least the number of shards")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"{args.prefix}_{index:03d}.jsonl.gz"
             for index in range(args.shards)]
    base, remainder = divmod(args.total_examples, args.shards)
    for shard_index, output in enumerate(paths):
        count = base + (1 if shard_index < remainder else 0)
        if count == 0:
            continue
        if output.exists():
            print(f"resume: completed shard {shard_index + 1}/{args.shards}: {output}", flush=True)
            continue
        command = [
            sys.executable, str(ROOT / "cache_teacher.py"),
            "--input", str(args.input), "--output", str(output),
            "--config", str(args.config), "--device", args.device,
            "--num-shards", str(args.shards), "--shard-index", str(shard_index),
            "--max-examples", str(count),
            "--teacher-batch-size", str(args.teacher_batch_size),
            "--score-micro-batch", str(args.score_micro_batch), "--resume",
        ]
        if args.index_corpus:
            command.extend(["--index-corpus", str(args.index_corpus)])
        if args.model_id:
            command.extend(["--model-id", args.model_id])
        if args.mode:
            command.extend(["--mode", args.mode])
        if args.index_max_rows:
            command.extend(["--index-max-rows", str(args.index_max_rows)])
        if args.mock_teacher:
            command.append("--mock-teacher")
        print(f"starting shard {shard_index + 1}/{args.shards}", flush=True)
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "1")
        subprocess.run(command, check=True, env=environment)

    completed = [path for path in paths if path.exists()]
    if len(completed) != len(paths):
        raise RuntimeError(f"only {len(completed)}/{len(paths)} shards completed; rerun this command")
    if args.merge_output:
        if args.merge_output.exists():
            print(f"resume: merged cache already exists: {args.merge_output}")
        else:
            rows = (row for path in paths for row in read_jsonl(path))
            count = write_jsonl(args.merge_output, rows, desc="merge teacher shards", atomic=True)
            print(f"merged {count:,} examples into {args.merge_output}")


if __name__ == "__main__":
    main()
