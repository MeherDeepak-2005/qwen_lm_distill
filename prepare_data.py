#!/usr/bin/env python3
"""Normalize one-utterance-per-line sources and make deterministic leakage-safe splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator

from utils import ROOT, stable_fraction


URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
ENGLISH = re.compile(r"[^a-z' ]+")
SPACES = re.compile(r"\s+")


def clean_english(text: str, min_words: int, max_words: int) -> str | None:
    text = URL.sub(" ", text.lower())
    text = ENGLISH.sub(" ", text)
    text = SPACES.sub(" ", text).strip()
    words = text.split()
    return text if min_words <= len(words) <= max_words else None


def source_lines(path: Path, text_key: str) -> Iterator[tuple[str, str]]:
    """Stream (group_id, text). JSONL group IDs prevent conversation leakage."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row[text_key]
                group = str(row.get("conversation_id", row.get("group_id", index)))
                yield group, text
    else:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                yield str(index), line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Stable source name, e.g. nus_sms")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--text-key", default="text", help="JSONL text field")
    parser.add_argument("--mode", choices=("en", "te", "xlit"), default="en")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--keep-fraction", type=float, default=1.0,
                        help="Deterministically retain this fraction of groups before cleaning")
    args = parser.parse_args()
    if not 0 < args.keep_fraction <= 1:
        raise ValueError("--keep-fraction must be in (0, 1]")

    seen: set[bytes] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: args.output_dir / f"{args.source}.{split}.jsonl"
             for split in ("train", "dev", "test")}
    temporary = {split: path.with_suffix(path.suffix + ".tmp") for split, path in paths.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    streams = {split: path.open("w", encoding="utf-8") for split, path in temporary.items()}
    counts = {split: 0 for split in paths}
    dropped = sampled_out = scanned = 0
    try:
        for group, raw in source_lines(args.input, args.text_key):
            scanned += 1
            if stable_fraction(f"sample:{args.source}:{group}", args.seed) >= args.keep_fraction:
                sampled_out += 1
                continue
            text = (clean_english(raw, args.min_words, args.max_words) if args.mode == "en"
                    else SPACES.sub(" ", raw).strip())
            if not text:
                dropped += 1
                continue
            digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            fraction = stable_fraction(f"{args.source}:{group}", args.seed)
            split = "test" if fraction < args.test_fraction else (
                "dev" if fraction < args.test_fraction + args.dev_fraction else "train"
            )
            row = {"mode": args.mode, "source": args.source,
                   "group_id": group, "text": text}
            streams[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    for split, path in paths.items():
        os.replace(temporary[split], path)
    print(json.dumps({"source": args.source, "counts": counts, "scanned": scanned,
                      "sampled_out": sampled_out, "dropped": dropped,
                      "keep_fraction": args.keep_fraction}, indent=2))


if __name__ == "__main__":
    main()
