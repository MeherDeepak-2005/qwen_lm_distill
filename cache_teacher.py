#!/usr/bin/env python3
"""Build tokenizer-independent Qwen soft targets over trie/ngram/Qwen word candidates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm

from candidates import BigramCandidates, WORD, WordTrie, corrupt_word, merge_candidates
from runtime import (DEVICE_CHOICES, configure_device, peak_memory_mb,
                     print_device_report, resolve_device)
from utils import ROOT, load_config, read_jsonl, write_jsonl


def build_indexes(corpus: Path, max_words: int = 100_000,
                  max_rows: int = 2_000_000) -> tuple[WordTrie, BigramCandidates]:
    frequencies: Counter[str] = Counter()
    ngrams = BigramCandidates()
    progress = tqdm(read_jsonl(corpus), total=max_rows, desc="build trie/ngram index",
                    unit="row", dynamic_ncols=True, mininterval=0.5)
    for row_index, row in enumerate(progress):
        if row_index >= max_rows:
            break
        words = [word.lower() for word in WORD.findall(row["text"])]
        frequencies.update(words)
        ngrams.observe(row["text"])
    progress.close()
    print(f"index ready: {len(frequencies):,} unique words", flush=True)
    trie = WordTrie()
    for word, frequency in frequencies.most_common(max_words):
        trie.insert(word, frequency)
    return trie, ngrams


class MockTeacher:
    """Deterministic no-download teacher for pipeline tests, never for real training."""
    def generate(self, context: str, limit: int) -> list[str]:
        return []

    def score(self, context: str, candidates: list[str]) -> list[float]:
        return [-math.log(index + 2) - 0.05 * len(word) for index, word in enumerate(candidates)]

    def generate_batch(self, contexts: list[str], limit: int) -> list[list[str]]:
        return [self.generate(context, limit) for context in contexts]

    def score_batch(self, contexts: list[str], candidate_lists: list[list[str]],
                    micro_batch: int) -> list[list[float]]:
        return [self.score(context, candidates)
                for context, candidates in zip(contexts, candidate_lists)]


class QwenTeacher:
    def __init__(self, model_id: str, device: torch.device, dtype: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype == "auto":
            dtype = ("bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported()
                     else "float16" if device.type in ("cuda", "mps") else "float32")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        self.device = device
        self.dtype_name = dtype
        print(f"loading teacher tokenizer: {model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("teacher tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print(f"loading teacher model on {device} as {dtype}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        print("teacher model ready", flush=True)

    @staticmethod
    def first_word(text: str) -> str | None:
        match = WORD.search(text.strip())
        return match.group(0).lower() if match else None

    @torch.inference_mode()
    def generate(self, context: str, limit: int) -> list[str]:
        return self.generate_batch([context], limit)[0]

    @torch.inference_mode()
    def generate_batch(self, contexts: list[str], limit: int) -> list[list[str]]:
        encoded = self.tokenizer(contexts, padding=True, return_tensors="pt").to(self.device)
        sequences = self.model.generate(
            **encoded, max_new_tokens=6, num_beams=limit, num_return_sequences=limit,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
        )
        prefix = encoded["input_ids"].shape[1]
        result: list[list[str]] = []
        for context_index in range(len(contexts)):
            words: list[str] = []
            start = context_index * limit
            for sequence in sequences[start:start + limit]:
                word = self.first_word(self.tokenizer.decode(sequence[prefix:], skip_special_tokens=True))
                if word and word not in words:
                    words.append(word)
            result.append(words)
        return result

    @torch.inference_mode()
    def score(self, context: str, candidates: list[str]) -> list[float]:
        return self.score_batch([context], [candidates], len(candidates))[0]

    @torch.inference_mode()
    def score_batch(self, contexts: list[str], candidate_lists: list[list[str]],
                    micro_batch: int) -> list[list[float]]:
        """Score complete words in bounded GPU batches without materializing FP32 logits for all positions."""
        rows: list[list[int]] = []
        starts: list[tuple[int, int]] = []
        owners: list[tuple[int, int]] = []
        for record_index, (context, candidates) in enumerate(zip(contexts, candidate_lists)):
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            for candidate_index, word in enumerate(candidates):
                continuation = self.tokenizer.encode(" " + word, add_special_tokens=False)
                if not continuation:
                    continuation = [self.tokenizer.unk_token_id]
                rows.append(context_ids + continuation)
                starts.append((len(context_ids), len(continuation)))
                owners.append((record_index, candidate_index))

        output = [[0.0] * len(candidates) for candidates in candidate_lists]
        pad = self.tokenizer.pad_token_id
        for offset in range(0, len(rows), micro_batch):
            chunk_rows = rows[offset:offset + micro_batch]
            chunk_starts = starts[offset:offset + micro_batch]
            max_length = max(map(len, chunk_rows))
            input_ids = torch.full((len(chunk_rows), max_length), pad,
                                   dtype=torch.long, device=self.device)
            attention = torch.zeros_like(input_ids)
            for index, row in enumerate(chunk_rows):
                input_ids[index, :len(row)] = torch.tensor(row, device=self.device)
                attention[index, :len(row)] = 1
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits

            batch_indices, positions, targets, score_owner = [], [], [], []
            for batch_index, (start, length) in enumerate(chunk_starts):
                for continuation_offset in range(length):
                    token_position = start + continuation_offset
                    batch_indices.append(batch_index)
                    positions.append(token_position - 1)
                    targets.append(input_ids[batch_index, token_position])
                    score_owner.append(batch_index)
            selected = logits[
                torch.tensor(batch_indices, device=self.device),
                torch.tensor(positions, device=self.device),
            ].float()
            target_ids = torch.stack(targets).long()
            token_scores = (selected.gather(1, target_ids[:, None]).squeeze(1)
                            - torch.logsumexp(selected, dim=-1))
            local_scores = [0.0] * len(chunk_rows)
            for owner, score in zip(score_owner, token_scores.tolist()):
                local_scores[owner] += score
            for local_index, score in enumerate(local_scores):
                record_index, candidate_index = owners[offset + local_index]
                output[record_index][candidate_index] = score
            del logits, selected, input_ids, attention
        return output


def examples(path: Path, max_context_words: int, seed: int):
    for row_index, row in enumerate(read_jsonl(path)):
        words = WORD.findall(row["text"].lower())
        for position in range(2, len(words)):
            target = words[position]
            if len(target) < 2:
                continue
            context = " ".join(words[max(0, position - max_context_words):position])
            key = f"{row.get('source')}:{row.get('group_id')}:{row_index}:{position}"
            digest = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
            yield row, context, target, corrupt_word(target, digest), digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index-corpus", type=Path, help="Defaults to input")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "teacher_targets.jsonl.gz")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=DEVICE_CHOICES)
    parser.add_argument("--model-id")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--teacher-batch-size", type=int)
    parser.add_argument("--score-micro-batch", type=int)
    parser.add_argument("--index-max-rows", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--mock-teacher", action="store_true", help="Smoke tests only")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")

    cfg = load_config(args.config)
    teacher_cfg = cfg["teacher"]
    device = resolve_device(args.device or teacher_cfg["device"])
    configure_device(device)
    print_device_report(device, teacher_cfg["dtype"])
    index_max_rows = args.index_max_rows or teacher_cfg.get("index_max_rows", 2_000_000)
    trie, ngrams = build_indexes(args.index_corpus or args.input, max_rows=index_max_rows)
    teacher_model_id = args.model_id or teacher_cfg["model_id"]
    teacher = MockTeacher() if args.mock_teacher else QwenTeacher(
        teacher_model_id, device, teacher_cfg["dtype"],
    )
    limit = teacher_cfg["candidate_limit"]
    max_examples = args.max_examples or teacher_cfg["max_examples"]
    teacher_batch_size = args.teacher_batch_size or teacher_cfg.get("batch_size", 1)
    score_micro_batch = args.score_micro_batch or teacher_cfg.get("score_micro_batch", 16)
    if teacher_batch_size < 1 or score_micro_batch < 1:
        raise ValueError("teacher batch sizes must be positive")

    selected = (
        item for item in examples(args.input, teacher_cfg["max_context_words"], cfg["training"]["seed"])
        if item[4] % args.num_shards == args.shard_index
    )

    def rows():
        count = 0
        progress = tqdm(total=max_examples, desc="cache teacher targets", unit="example",
                        dynamic_ncols=True, mininterval=0.5)
        try:
            while count < max_examples:
                batch = list(itertools.islice(selected, min(teacher_batch_size, max_examples - count)))
                if not batch:
                    break
                contexts = [item[1] for item in batch]
                try:
                    qwen_batches = teacher.generate_batch(contexts, teacher_cfg["qwen_candidates"])
                    candidate_lists, source_lists = [], []
                    for (_, context, target, typed, _), qwen in zip(batch, qwen_batches):
                        trie_words = trie.search(typed, max_distance=2, limit=teacher_cfg["trie_candidates"])
                        ngram_words = ngrams.next_words(context, teacher_cfg["ngram_candidates"])
                        candidates, sources = merge_candidates(
                            target, {"trie": trie_words, "ngram": ngram_words, "qwen": qwen}, limit,
                        )
                        candidate_lists.append(candidates)
                        source_lists.append(sources)
                    score_lists = teacher.score_batch(contexts, candidate_lists, score_micro_batch)
                except torch.cuda.OutOfMemoryError as error:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "teacher CUDA OOM; reduce --teacher-batch-size or --score-micro-batch"
                    ) from error
                for item, candidates, sources, scores in zip(batch, candidate_lists, source_lists, score_lists):
                    source_row, context, target, typed, _ = item
                    yield {
                        "mode": source_row.get("mode", "en"), "context": context,
                        "typed": typed, "target": target, "candidates": candidates,
                        "teacher_log_probs": [round(score, 6) for score in scores],
                        "candidate_sources": sources, "source": source_row.get("source", "unknown"),
                    }
                    count += 1
                    progress.update(1)
        finally:
            progress.close()

    count = write_jsonl(args.output, rows())
    print(json.dumps({"output": str(args.output), "examples": count,
                      "teacher": "mock" if args.mock_teacher else teacher_model_id,
                      "teacher_dtype": getattr(teacher, "dtype_name", None),
                      "shard_index": args.shard_index, "num_shards": args.num_shards,
                      "teacher_batch_size": teacher_batch_size,
                      "score_micro_batch": score_micro_batch,
                      "index_max_rows": index_max_rows,
                      "peak_device_mb": peak_memory_mb(device)}, indent=2))


if __name__ == "__main__":
    main()
