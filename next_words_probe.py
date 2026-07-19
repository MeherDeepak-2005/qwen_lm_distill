#!/usr/bin/env python3
"""Production-equivalent n-gram vs neural next-word probe for the 10.6M student."""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path

import torch

from probe_common import DEFAULT_LEGACY_TOOLS, StudentProbe, load_keyboard_harness
from utils import ROOT


BEAM_WIDTH = 8
MAX_PIECES = 4
LIMIT = 8
TOPK_PER_STEP = 64


def trie_contains(root, surface: str, complete: bool = False) -> bool:
    node = root
    for character in surface:
        node = node.get(character)
        if node is None:
            return False
    return "$" in node if complete else True


def logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


class NextWordModels:
    def __init__(self, checkpoint: Path, tokenizer: Path, legacy_tools: Path,
                 langdata: Path | None, device: str):
        print("loading production langdata.bin + trie ...")
        self.harness, self.ld, self.trie = load_keyboard_harness(legacy_tools, langdata)
        print(f"loading 10.6M student: {checkpoint}")
        self.student = StudentProbe(checkpoint, tokenizer, device)
        self.sp = self.student.sp
        self.boundary_ids = [
            index for index in range(self.sp.vocab_size())
            if self.sp.id_to_piece(index).startswith("▁")
        ]
        if self.sp.eos_id() >= 0:
            self.boundary_ids.append(self.sp.eos_id())

    def ngram_next_words(self, context: list[str], limit: int = LIMIT) -> list[str]:
        context = [word.lower() for word in context]
        if len(context) >= 2:
            left, right = self.ld.ids.get(context[-2]), self.ld.ids.get(context[-1])
            if left is not None and right is not None:
                hit = self.ld.trigrams(left, right)
                if hit:
                    words = [self.ld.word(word) for word, _ in sorted(hit[1], key=lambda row: -row[1])]
                    if words:
                        return words[:limit]
        if context:
            word_id = self.ld.ids.get(context[-1])
            if word_id is not None and (hit := self.ld.bigrams(word_id)):
                words = [self.ld.word(word) for word, _ in sorted(hit[1], key=lambda row: -row[1])]
                if words:
                    return words[:limit]
        order = sorted(range(self.ld.W), key=lambda index: -self.ld.freqs[index])
        return [self.ld.word(index) for index in order[:limit]]

    def neural_next_words(self, context: list[str], limit: int = LIMIT,
                          beam_width: int = BEAM_WIDTH, max_pieces: int = MAX_PIECES):
        context_ids = self.student.context_ids(context)
        context_ids = context_ids[-(self.student.max_length - max_pieces):]
        beams = [{"tokens": context_ids, "surface": "", "score": 0.0}]
        found: dict[str, float] = {}
        root = self.trie.root
        for _ in range(max_pieces):
            distributions = self.student.last_token_logprobs([beam["tokens"] for beam in beams])
            expanded = []
            for beam, distribution in zip(beams, distributions):
                if beam["surface"] and trie_contains(root, beam["surface"], complete=True):
                    ending = -math.inf
                    for boundary in self.boundary_ids:
                        ending = logaddexp(ending, distribution[boundary].item())
                    found[beam["surface"]] = max(
                        found.get(beam["surface"], -math.inf), beam["score"] + ending,
                    )
                values, indices = torch.topk(distribution, min(TOPK_PER_STEP, distribution.numel()))
                for log_probability, token_id in zip(values.tolist(), indices.tolist()):
                    piece = self.sp.id_to_piece(token_id)
                    if not piece or piece.startswith("<"):
                        continue
                    if not beam["surface"]:
                        if not piece.startswith("▁") or len(piece) == 1:
                            continue
                        surface = piece[1:]
                    else:
                        if piece.startswith("▁"):
                            continue
                        surface = beam["surface"] + piece
                    if trie_contains(root, surface):
                        expanded.append({"tokens": beam["tokens"] + [token_id],
                                         "surface": surface,
                                         "score": beam["score"] + log_probability})
            beams = heapq.nlargest(beam_width, expanded, key=lambda beam: beam["score"])
            if not beams:
                break
        return sorted(found.items(), key=lambda row: -row[1])[:limit]


def run(models: NextWordModels, context_text: str) -> None:
    context = context_text.strip().split()
    ngram = models.ngram_next_words(context)
    neural = models.neural_next_words(context)
    print(f"\ncontext={context!r}")
    print(f"{'#':>2}  {'n-gram':30}  {'10.6M neural (score)':30}")
    for index in range(max(len(ngram), len(neural))):
        left = ngram[index] if index < len(ngram) else ""
        right = f"{neural[index][0]} ({neural[index][1]:.2f})" if index < len(neural) else ""
        print(f"{index + 1:>2}  {left:30}  {right:30}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context", nargs="*")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "pretrain.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "spm.model")
    parser.add_argument("--legacy-tools", type=Path, default=DEFAULT_LEGACY_TOOLS)
    parser.add_argument("--langdata", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    models = NextWordModels(args.checkpoint, args.tokenizer, args.legacy_tools,
                            args.langdata, args.device)
    if args.context:
        run(models, " ".join(args.context))
        return
    print("\nType a context. Blank line quits.")
    try:
        while line := input("\ncontext> "):
            run(models, line)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
