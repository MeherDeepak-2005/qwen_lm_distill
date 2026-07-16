#!/usr/bin/env python3
"""Measure the exported fixed-batch Core ML graph on the current Mac."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--graph", choices=("next", "candidate"), default="next")
    args = parser.parse_args()
    import coremltools as ct

    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.ALL)
    rng = np.random.default_rng(17)

    def inputs():
        batch = 1 if args.graph == "next" else 16
        tokens = rng.integers(4, 6000, size=(batch, 32), dtype=np.int32)
        value = {
            "tokens": tokens,
            "next_positions": np.asarray([[31]], dtype=np.int32),
        }
        if args.graph == "candidate":
            value.update({
                "score_positions": np.tile(np.arange(8, dtype=np.int32), (16, 1)),
                "target_ids": rng.integers(4, 6000, size=(16, 8), dtype=np.int32),
                "score_mask": np.ones((16, 8), dtype=np.float32),
            })
        return value

    for _ in range(args.warmup):
        model.predict(inputs())
    elapsed = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        model.predict(inputs())
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    result = {
        "model": str(args.model), "package_mb": package_bytes(args.model) / 1_000_000,
        "iterations": args.iterations, "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)), "p99_ms": float(np.percentile(elapsed, 99)),
        "mean_ms": float(np.mean(elapsed)),
        "note": "Mac Core ML timing is a preflight; measure the keyboard extension on the target iPhone before shipping.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
