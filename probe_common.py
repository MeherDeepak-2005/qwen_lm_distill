"""Shared bridge from the 10.6M student to the keyboard's production test harness."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from inference import load_student, score_record
from runtime import resolve_device
from utils import ROOT, tokenizer_hash


DEFAULT_LEGACY_TOOLS = Path(os.environ.get(
    "KEYBOARD_LM_TOOLS", "/Users/meher/Downloads/xcode-projects/keyboard/tools/lm",
))


def load_keyboard_harness(tools_dir: Path = DEFAULT_LEGACY_TOOLS,
                          langdata: Path | None = None):
    tools_dir = tools_dir.expanduser().resolve()
    harness_path = tools_dir / "eval_harness.py"
    if not harness_path.exists():
        raise FileNotFoundError(
            f"missing {harness_path}; pass --legacy-tools or set KEYBOARD_LM_TOOLS"
        )
    sys.path.insert(0, str(tools_dir))
    harness = importlib.import_module("eval_harness")
    path = (langdata or harness.LANGDATA).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing keyboard lexicon: {path}")
    lexicon = harness.LangData(path)
    return harness, lexicon, harness.Trie(lexicon)


class StudentProbe:
    def __init__(self, checkpoint: Path = ROOT / "artifacts" / "pretrain.pt",
                 tokenizer: Path = ROOT / "artifacts" / "spm.model",
                 device_name: str = "auto"):
        self.device = resolve_device(device_name)
        self.tokenizer_path = tokenizer.expanduser().resolve()
        self.sp = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))
        self.model, self.checkpoint = load_student(checkpoint.expanduser().resolve(), self.device)
        expected = self.checkpoint.get("tokenizer_sha256")
        if expected and expected != tokenizer_hash(self.tokenizer_path):
            raise RuntimeError("checkpoint/tokenizer hash mismatch")
        self.max_length = self.model.config.max_length

    def context_ids(self, context: list[str], mode: str = "en") -> list[int]:
        tag = {"en": "<en>", "te": "<te>", "xlit": "<xlit>"}[mode]
        return [self.sp.piece_to_id(tag)] + self.sp.encode(" ".join(context).lower(), out_type=int)

    def score_words(self, context: list[str], words: list[str], mode: str = "en") -> list[float]:
        if not words:
            return []
        record = {
            "mode": mode, "context": " ".join(context).lower(),
            "candidates": words, "target": words[0],
        }
        return score_record(self.model, record, self.sp, self.device).scores

    @torch.inference_mode()
    def last_token_logprobs(self, rows: list[list[int]]) -> list[torch.Tensor]:
        if not rows:
            return []
        rows = [row[-self.max_length:] for row in rows]
        length = max(len(row) for row in rows)
        batch = torch.full((len(rows), length), self.sp.pad_id(), dtype=torch.long,
                           device=self.device)
        positions = []
        for index, row in enumerate(rows):
            batch[index, :len(row)] = torch.tensor(row, device=self.device)
            positions.append(len(row) - 1)
        logits = self.model(batch)
        return [torch.log_softmax(logits[index, position].float(), dim=-1).cpu()
                for index, position in enumerate(positions)]
