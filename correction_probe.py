#!/usr/bin/env python3
"""Production-equivalent correction probe using the new 10.6M student checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from probe_common import DEFAULT_LEGACY_TOOLS, StudentProbe, load_keyboard_harness
from utils import ROOT


W_NEURAL, W_NGRAM = 0.7, 0.3
CORRECTION_MARGIN = 3.0
REAL_WORD_MARGIN = 5.0
REAL_WORD_MAX_COST = 1.6
TOPN = 15


class CorrectionModels:
    def __init__(self, checkpoint: Path, tokenizer: Path, legacy_tools: Path,
                 langdata: Path | None, device: str):
        print("loading production langdata.bin + trie ...")
        self.harness, self.ld, self.trie = load_keyboard_harness(legacy_tools, langdata)
        print(f"loading 10.6M student: {checkpoint}")
        self.student = StudentProbe(checkpoint, tokenizer, device)


def apostrophe_restoration(ld, word: str) -> str | None:
    if "'" in word or len(word) < 2:
        return None
    best = None
    for index in range(1, len(word)):
        candidate = word[:index] + "'" + word[index:]
        word_id = ld.ids.get(candidate)
        if word_id is not None and (best is None or ld.freqs[word_id] > best[1]):
            best = candidate, ld.freqs[word_id]
    return best[0] if best else None


def build_table(models: CorrectionModels, typed: str, context: list[str], budget: float | None):
    typed = typed.lower()
    context = [word.lower() for word in context]
    is_real_word = typed in models.ld.ids
    used_budget = budget if budget is not None else models.harness.cost_budget(len(typed))
    candidates = models.trie.search(typed, max_cost=used_budget)
    candidates.sort(key=lambda item: -item[1] + models.harness.FREQ_W * item[2], reverse=True)
    kept = candidates[:TOPN]
    if any(word == typed for word, _, _ in kept):
        sources = kept
    else:
        word_id = models.ld.ids.get(typed)
        log_frequency = math.log(models.ld.freqs[word_id] + 1) if word_id is not None else 0.0
        sources = kept + [(typed, 0.0, log_frequency)]

    words = [word for word, _, _ in sources]
    neural_scores = models.student.score_words(context, words)
    rows = []
    for (word, cost, log_frequency), neural in zip(sources, neural_scores):
        ngram = models.harness.logprob(models.ld, word, context)
        blend = W_NEURAL * neural + W_NGRAM * ngram
        final = -cost + models.harness.LM_W * blend + models.harness.FREQ_W * log_frequency
        rows.append({"word": word, "cost": cost, "neural": neural, "ngram": ngram,
                     "blend": blend, "final": final})
    rows.sort(key=lambda row: -row["final"])
    return rows, is_real_word, used_budget, len(candidates)


def gate(rows: list[dict], typed: str, is_real_word: bool):
    typed_row = next(row for row in rows if row["word"] == typed)
    best = rows[0]
    if is_real_word:
        threshold = REAL_WORD_MARGIN
        eligible = best["word"] != typed and best["cost"] <= REAL_WORD_MAX_COST
        rule = f"real-word: margin > {threshold} and cost <= {REAL_WORD_MAX_COST}"
    else:
        threshold = CORRECTION_MARGIN
        eligible = best["word"] != typed
        rule = f"OOV: margin > {threshold}"
    margin = best["final"] - typed_row["final"]
    return margin, eligible and margin > threshold, rule, best["word"]


def run(models: CorrectionModels, typed: str, context: list[str], budget: float | None):
    typed = typed.lower()
    if typed not in models.ld.ids:
        restored = apostrophe_restoration(models.ld, typed)
        if restored:
            print(f"\n{typed!r} -> {restored!r}  (apostrophe restoration)")
            return
    rows, real_word, used_budget, generated = build_table(models, typed, context, budget)
    print(f"\ntyped={typed!r}  context={context!r}  budget={used_budget:.2f}  candidates={generated}")
    print(f"{'candidate':14} {'channel':>8} {'neural':>9} {'n-gram':>9} {'blend':>9} {'final':>9}")
    for row in rows[:12]:
        marker = "  <- typed" if row["word"] == typed else ""
        print(f"{row['word']:14} {-row['cost']:8.2f} {row['neural']:9.2f} "
              f"{row['ngram']:9.2f} {row['blend']:9.2f} {row['final']:9.2f}{marker}")
    margin, fire, rule, winner = gate(rows, typed, real_word)
    print(f"\n{rule}; best margin={margin:.2f}")
    print(f"=> {'CORRECT to ' + repr(winner) if fire else 'keep typed'}")


def parse_line(line: str):
    parts = [part.strip() for part in line.split("|")]
    return (parts[0], parts[1].split() if len(parts) > 1 and parts[1] else [],
            float(parts[2]) if len(parts) > 2 and parts[2] else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("typed", nargs="?")
    parser.add_argument("--context", default="")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "pretrain.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "spm.model")
    parser.add_argument("--legacy-tools", type=Path, default=DEFAULT_LEGACY_TOOLS)
    parser.add_argument("--langdata", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    models = CorrectionModels(args.checkpoint, args.tokenizer, args.legacy_tools,
                              args.langdata, args.device)
    if args.typed:
        run(models, args.typed, args.context.split(), args.budget)
        return
    print("\nType: word | optional context | optional budget. Blank line quits.")
    try:
        while line := input("\ncorrection> "):
            run(models, *parse_line(line))
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
