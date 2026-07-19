#!/usr/bin/env python3
"""Gate simulated per-channel int8 weights against FP32 word-ranking quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from tqdm.auto import tqdm

from inference import load_student, ranking_metrics, score_record
from runtime import DEVICE_CHOICES, print_device_report, resolve_device
from utils import ROOT, load_config, read_jsonl, tokenizer_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "spm.model")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--no-enforce", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output and args.output.exists():
        print(f"resume: quantization verification already completed: {args.output}")
        return

    device = resolve_device(args.device)
    print_device_report(device)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, checkpoint = load_student(args.checkpoint, device)
    if checkpoint.get("tokenizer_sha256") != tokenizer_hash(args.tokenizer):
        raise RuntimeError("checkpoint/tokenizer hash mismatch")
    records = []
    for row in tqdm(read_jsonl(args.input), total=args.max_examples,
                    desc="load quantization records", unit="row", dynamic_ncols=True):
        records.append(row)
        if len(records) >= args.max_examples:
            break
    if not records:
        raise RuntimeError("empty quantization evaluation set")

    fp32 = [score_record(model, row, tokenizer, device) for row in tqdm(
        records, desc="score FP32/QAT-master", unit="row", dynamic_ncols=True,
    )]
    model.enable_qat(True)
    int8 = [score_record(model, row, tokenizer, device) for row in tqdm(
        records, desc="score simulated int8", unit="row", dynamic_ncols=True,
    )]
    fp_metrics, int8_metrics = ranking_metrics(fp32), ranking_metrics(int8)
    fp_top = [int(np.argmax(row.scores)) for row in fp32]
    int8_top = [int(np.argmax(row.scores)) for row in int8]
    agreement = sum(a == b for a, b in zip(fp_top, int8_top)) / len(fp_top)
    flat_fp = np.asarray([score for row in fp32 for score in row.scores], dtype=np.float64)
    flat_int8 = np.asarray([score for row in int8 for score in row.scores], dtype=np.float64)
    correlation = float(np.corrcoef(flat_fp, flat_int8)[0, 1]) if flat_fp.size > 1 else 1.0
    if not math.isfinite(correlation):
        correlation = 1.0 if np.allclose(flat_fp, flat_int8) else 0.0

    gates = load_config(args.config)["quantization"]
    checks = {
        "top3_absolute_drop": fp_metrics["top3"] - int8_metrics["top3"] <= gates["max_top3_absolute_drop"],
        "mrr_relative_drop": (fp_metrics["mrr"] - int8_metrics["mrr"]) / max(fp_metrics["mrr"], 1e-9)
                             <= gates["max_mrr_relative_drop"],
        "top1_agreement": agreement >= gates["min_top1_agreement"],
        "score_correlation": correlation >= gates["min_score_correlation"],
    }
    result = {"fp32": fp_metrics, "simulated_int8": int8_metrics,
              "top1_agreement": agreement, "score_correlation": correlation,
              "checks": checks, "passed": all(checks.values()),
              "warning": "This gates weight quantization numerically. Run export_coreml.py --smoke-predict to also test the deployed Core ML graph."}
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n")
        temporary.replace(args.output)
    if not result["passed"] and not args.no_enforce:
        raise SystemExit("int8 quality gate failed; run QAT and re-evaluate")


if __name__ == "__main__":
    main()
