from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch

from utils import read_jsonl


MODE_TAG = {"en": "<en>", "te": "<te>", "xlit": "<xlit>"}


def shuffled_rows(path: Path, rng: random.Random, buffer_size: int = 10_000):
    """Bounded-memory shuffle that cycles forever for token-budget training."""
    while True:
        buffer = []
        for row in read_jsonl(path):
            if len(buffer) < buffer_size:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer
        if not buffer:
            raise RuntimeError(f"empty dataset: {path}")


def packed_batches(path: Path, tokenizer, batch_size: int, max_length: int,
                   seed: int = 17) -> Iterator[torch.Tensor]:
    rng = random.Random(seed)
    stream = shuffled_rows(path, rng)
    token_buffer: list[int] = []
    sequence_length = max_length + 1
    while True:
        rows = []
        while len(rows) < batch_size:
            while len(token_buffer) < sequence_length:
                row = next(stream)
                tag = MODE_TAG[row.get("mode", "en")]
                token_buffer.extend(tokenizer.encode(f"{tag} {row['text']}", out_type=int))
                token_buffer.append(tokenizer.eos_id())
            rows.append(token_buffer[:sequence_length])
            del token_buffer[:sequence_length]
        yield torch.tensor(rows, dtype=torch.long)


def record_batches(path: Path, batch_size: int, seed: int = 17) -> Iterator[list[dict]]:
    rng = random.Random(seed)
    stream = shuffled_rows(path, rng, buffer_size=4096)
    while True:
        yield [next(stream) for _ in range(batch_size)]


def build_hard_batch(records: list[dict], tokenizer, max_length: int, device: torch.device):
    batch = len(records)
    pad = tokenizer.pad_id()
    inputs = torch.full((batch, max_length), pad, dtype=torch.long, device=device)
    labels = torch.full((batch, max_length), -100, dtype=torch.long, device=device)
    boundaries = torch.zeros(batch, dtype=torch.long, device=device)
    contexts: list[list[int]] = []
    targets: list[list[int]] = []

    for index, record in enumerate(records):
        tag_id = tokenizer.piece_to_id(MODE_TAG[record.get("mode", "en")])
        context = [tag_id] + tokenizer.encode(record["context"], out_type=int)
        target = tokenizer.encode(" " + record["target"], out_type=int)
        if not target:
            target = [tokenizer.unk_id()]
        context = context[-max(1, max_length + 1 - len(target)):]
        full = (context + target)[:max_length + 1]
        x, y = full[:-1], full[1:]
        inputs[index, :len(x)] = torch.tensor(x, device=device)
        target_start = max(0, len(context) - 1)
        labels[index, target_start:len(y)] = torch.tensor(y[target_start:], device=device)
        boundaries[index] = len(context) - 1
        contexts.append(context)
        targets.append(target)
    return inputs, labels, boundaries, contexts, targets
