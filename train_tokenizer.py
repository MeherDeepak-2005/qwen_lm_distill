#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm

from utils import ROOT, read_jsonl


class CyclingRows:
    def __init__(self, path: str):
        self.path = Path(path)
        self.rows = iter(read_jsonl(self.path))

    def next(self) -> dict:
        try:
            return next(self.rows)
        except StopIteration:
            self.rows = iter(read_jsonl(self.path))
            try:
                return next(self.rows)
            except StopIteration as error:
                raise RuntimeError(f"empty tokenizer source: {self.path}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Processed JSONL shards")
    parser.add_argument("--input-weights", nargs="+", type=float,
                        help="Sampling weights in the same order as --input; default is equal")
    parser.add_argument("--output-prefix", default=str(ROOT / "artifacts" / "spm"))
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--sample-lines", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    weights = args.input_weights or [1.0] * len(args.input)
    if len(weights) != len(args.input) or any(weight <= 0 for weight in weights):
        raise ValueError("--input-weights must contain one positive value per --input")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sample = prefix.parent / "tokenizer_sample.txt"
    streams = [CyclingRows(path) for path in args.input]
    rng = random.Random(args.seed)
    count = 0
    with sample.open("w", encoding="utf-8") as out:
        while count < args.sample_lines:
            stream = rng.choices(streams, weights=weights, k=1)[0]
            row = stream.next()
            mode = row.get("mode", "en")
            tag = {"en": "<en>", "te": "<te>", "xlit": "<xlit>"}[mode]
            out.write(f"{tag} {row['text']}\n")
            count += 1
    if count < 1000:
        raise RuntimeError(f"only {count} tokenizer lines; provide a real corpus")

    spm.SentencePieceTrainer.train(
        input=str(sample), model_prefix=str(prefix), model_type="unigram",
        vocab_size=args.vocab_size, character_coverage=0.9999, byte_fallback=True,
        split_by_unicode_script=False,
        user_defined_symbols=["<en>", "<te>", "<xlit>"],
        pad_id=3, unk_id=0, bos_id=1, eos_id=2,
        input_sentence_size=min(count, args.sample_lines), shuffle_input_sentence=True,
    )
    metadata = {"lines": count, "vocab_size": args.vocab_size,
                "inputs": args.input, "input_weights": weights,
                "future_modes": ["en", "te", "xlit"]}
    (prefix.parent / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("wrote", prefix.with_suffix(".model"))


if __name__ == "__main__":
    main()
