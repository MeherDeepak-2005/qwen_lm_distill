#!/usr/bin/env python3
"""Evaluate candidate-source recall and conditional student ranking quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from tqdm.auto import tqdm

from inference import load_student, ranking_metrics, score_record
from runtime import DEVICE_CHOICES, print_device_report, resolve_device
from utils import ROOT, read_jsonl, tokenizer_hash


def candidate_recall(records: list[dict]) -> dict[str, float]:
    sources = ("trie", "ngram", "qwen")
    hits = {source: 0 for source in sources}
    hits["union"] = 0
    for row in records:
        target_sources = set(row.get("candidate_sources", {}).get(row["target"], [])) - {"gold"}
        for source in sources:
            hits[source] += source in target_sources
        hits["union"] += bool(target_sources & set(sources))
    return {name: value / max(1, len(records)) for name, value in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Held-out teacher-target JSONL(.gz)")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "spm.model")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output and args.output.exists():
        print(f"resume: evaluation already completed: {args.output}")
        return

    device = resolve_device(args.device)
    print_device_report(device)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, checkpoint = load_student(args.checkpoint, device)
    if checkpoint.get("tokenizer_sha256") != tokenizer_hash(args.tokenizer):
        raise RuntimeError("checkpoint/tokenizer hash mismatch")
    records = []
    for row in tqdm(read_jsonl(args.input), total=args.max_examples,
                    desc="load evaluation records", unit="row", dynamic_ncols=True):
        records.append(row)
        if len(records) >= args.max_examples:
            break
    if not records:
        raise RuntimeError(f"no evaluation records in {args.input}")

    scored = [score_record(model, row, tokenizer, device) for row in tqdm(
        records, desc="score evaluation records", unit="row", dynamic_ncols=True,
    )]
    result = {
        "checkpoint": str(args.checkpoint),
        "candidate_recall": candidate_recall(records),
        "conditional_ranking": ranking_metrics(scored),
        "note": "Ranking is conditional on the cached candidate union; recall excludes forced gold insertion.",
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
