#!/usr/bin/env python3
"""Create a weighted, reproducible stage corpus without duplicating files on disk."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm.auto import tqdm

from utils import ROOT, load_config, read_jsonl, write_jsonl


class CyclingRows:
    def __init__(self, path: Path):
        self.path = path
        self.iterator = iter(read_jsonl(path))

    def next(self) -> dict:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(read_jsonl(self.path))
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise RuntimeError(f"empty source: {self.path}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-tokens", type=int, help="Whitespace-token exposure target")
    parser.add_argument("--resume", action="store_true",
                        help="Skip mixing when the final atomic output already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    key = "stage_a_weights" if args.stage == "a" else "stage_b_weights"
    target_key = "stage_a_target_tokens" if args.stage == "a" else "stage_b_target_tokens"
    weights: dict[str, float] = cfg["data"][key]
    target = args.target_tokens or cfg["data"][target_key]
    output = args.output or ROOT / "data" / "processed" / f"stage_{args.stage}_{args.split}.jsonl"
    if args.resume and output.exists():
        print(f"resume: already completed {output}")
        return
    missing = [name for name in weights if not (args.processed_dir / f"{name}.{args.split}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"missing normalized sources: {missing}")

    streams = {name: CyclingRows(args.processed_dir / f"{name}.{args.split}.jsonl") for name in weights}
    names, probabilities = list(weights), list(weights.values())
    rng = random.Random(cfg["data"]["seed"] + (0 if args.stage == "a" else 1))
    emitted_tokens = 0
    progress = tqdm(total=target, desc=f"mix stage {args.stage.upper()} {args.split}",
                    unit="tok", unit_scale=True, dynamic_ncols=True, mininterval=0.5)

    def rows():
        nonlocal emitted_tokens
        try:
            while emitted_tokens < target:
                name = rng.choices(names, weights=probabilities, k=1)[0]
                row = dict(streams[name].next())
                row["sampled_for"] = f"stage_{args.stage}_{args.split}"
                added = len(row["text"].split()) + 1
                emitted_tokens += added
                progress.update(added)
                yield row
        finally:
            progress.close()

    count = write_jsonl(output, rows(), atomic=True)
    print(json.dumps({"output": str(output), "lines": count, "approx_tokens": emitted_tokens,
                      "weights": weights}, indent=2))


if __name__ == "__main__":
    main()
