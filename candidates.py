"""Trie, n-gram and teacher candidate utilities plus source-ablation metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import regex as re


# Unicode letters with one optional apostrophe-delimited suffix. This supports
# English contractions, Telugu script, and Latin-script Tenglish.
WORD = re.compile(r"\p{L}[\p{L}\p{M}]*(?:['’]\p{L}[\p{L}\p{M}]*)?")


class TrieNode:
    __slots__ = ("children", "word", "frequency")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.word: str | None = None
        self.frequency = 0


class WordTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, frequency: int = 1) -> None:
        word = word.lower()
        if not WORD.fullmatch(word):
            return
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.word, node.frequency = word, max(node.frequency, frequency)

    def search(self, typed: str, max_distance: int = 2, limit: int = 8) -> list[str]:
        typed = typed.lower()
        initial = list(range(len(typed) + 1))
        found: list[tuple[int, int, str]] = []

        def visit(node: TrieNode, char: str, previous: list[int]) -> None:
            row = [previous[0] + 1]
            for column, target in enumerate(typed, 1):
                row.append(min(
                    row[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char != target),
                ))
            if node.word and row[-1] <= max_distance:
                found.append((row[-1], -node.frequency, node.word))
            if min(row) <= max_distance:
                for next_char, child in node.children.items():
                    visit(child, next_char, row)

        for char, child in self.root.children.items():
            visit(child, char, initial)
        found.sort()
        return [word for _, _, word in found[:limit]]


class BigramCandidates:
    def __init__(self):
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.unigrams: Counter[str] = Counter()

    def observe(self, text: str) -> None:
        words = [word.lower() for word in WORD.findall(text)]
        self.unigrams.update(words)
        for left, right in zip(words, words[1:]):
            self.counts[left][right] += 1

    def next_words(self, context: str, limit: int = 8) -> list[str]:
        words = WORD.findall(context.lower())
        counter = self.counts.get(words[-1], Counter()) if words else self.unigrams
        return [word for word, _ in counter.most_common(limit)]


def corrupt_word(word: str, seed_value: int) -> str:
    """Deterministic one-edit typo used to measure trie candidate recall."""
    if len(word) < 3:
        return word
    operation = seed_value % (3 if word.isascii() else 2)
    position = (seed_value // 3) % len(word)
    if operation == 0 and len(word) > 3:
        return word[:position] + word[position + 1:]
    if operation == 1 and position + 1 < len(word):
        chars = list(word)
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
        return "".join(chars)
    keyboard_neighbor = {"a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i"}
    replacement = keyboard_neighbor.get(word[position], "e" if word[position] != "e" else "r")
    return word[:position] + replacement + word[position + 1:]


def merge_candidates(target: str, groups: dict[str, Iterable[str]], limit: int) -> tuple[list[str], dict[str, list[str]]]:
    sources: dict[str, set[str]] = defaultdict(set)
    ordered = [target.lower()]
    sources[target.lower()].add("gold")
    for source, words in groups.items():
        for word in words:
            word = word.lower().strip()
            if not WORD.fullmatch(word):
                continue
            sources[word].add(source)
            if word not in ordered:
                ordered.append(word)
    ordered = ordered[:limit]
    return ordered, {word: sorted(sources[word]) for word in ordered}
