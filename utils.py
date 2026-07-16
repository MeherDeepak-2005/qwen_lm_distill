from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.json"
    return json.loads(path.read_text())


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_fraction(value: str, seed: int = 17) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
    except ImportError:
        pass


def cosine_lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def tokenizer_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
