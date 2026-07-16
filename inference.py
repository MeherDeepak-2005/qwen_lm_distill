"""Checkpoint loading and exact whole-word candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from datasets import MODE_TAG
from model import KeyboardDecoder, ModelConfig


def load_student(checkpoint_path, device: torch.device) -> tuple[KeyboardDecoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = KeyboardDecoder(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@dataclass
class ScoredRecord:
    scores: list[float]
    candidates: list[str]
    target: str


@torch.inference_mode()
def score_record(model: KeyboardDecoder, record: dict, tokenizer, device: torch.device) -> ScoredRecord:
    """Score complete candidate words, including every SentencePiece piece."""
    cfg = model.config
    tag = MODE_TAG[record.get("mode", "en")]
    context = [tokenizer.piece_to_id(tag)] + tokenizer.encode(record["context"], out_type=int)
    rows: list[tuple[list[int], list[int], int]] = []
    for candidate in record["candidates"]:
        pieces = tokenizer.encode(" " + candidate, out_type=int) or [tokenizer.unk_id()]
        kept_context = context[-max(1, cfg.max_length + 1 - len(pieces)):]
        full = (kept_context + pieces)[:cfg.max_length + 1]
        rows.append((full[:-1], full[1:], len(kept_context) - 1))

    inputs = torch.full((len(rows), cfg.max_length), tokenizer.pad_id(), dtype=torch.long, device=device)
    for index, (tokens, _, _) in enumerate(rows):
        inputs[index, :len(tokens)] = torch.tensor(tokens, device=device)
    log_probs = model(inputs).float().log_softmax(dim=-1)
    scores: list[float] = []
    for index, (_, labels, start) in enumerate(rows):
        total = torch.zeros((), device=device)
        for position in range(start, len(labels)):
            total += log_probs[index, position, labels[position]]
        scores.append(total.item())
    return ScoredRecord(scores, record["candidates"], record["target"])


def ranking_metrics(scored: list[ScoredRecord]) -> dict[str, float | int]:
    top1 = top3 = 0
    reciprocal_rank = 0.0
    for row in scored:
        order = sorted(range(len(row.scores)), key=row.scores.__getitem__, reverse=True)
        gold = row.candidates.index(row.target)
        rank = order.index(gold) + 1
        top1 += rank == 1
        top3 += rank <= 3
        reciprocal_rank += 1.0 / rank
    count = len(scored)
    return {
        "examples": count,
        "top1": top1 / max(1, count),
        "top3": top3 / max(1, count),
        "mrr": reciprocal_rank / max(1, count),
    }
