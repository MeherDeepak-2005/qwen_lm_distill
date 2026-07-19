#!/usr/bin/env python3
"""Convert common downloaded corpus layouts to leakage-aware JSONL."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import write_jsonl


def plain_rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield {"conversation_id": str(index), "text": line.strip()}


def dailydialog_rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as stream:
        for conversation, line in enumerate(stream):
            for turn, text in enumerate(line.split("__eou__")):
                if text.strip():
                    yield {"conversation_id": str(conversation), "turn": turn, "text": text.strip()}


def dailydialog_jsonl_rows(path: Path):
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for json_path in paths:
        with json_path.open(encoding="utf-8", errors="replace") as stream:
            for conversation, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                group = f"{json_path.stem}:{conversation}"
                for turn, utterance in enumerate(row.get("dialogue", [])):
                    text = utterance.get("text", "")
                    if isinstance(text, str) and text.strip():
                        yield {"conversation_id": group, "turn": turn, "text": text.strip()}


def gutenberg_rows(path: Path):
    """The published files separate dialogues with blank lines."""
    conversation = turn = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                if turn:
                    conversation += 1
                    turn = 0
                continue
            yield {"conversation_id": str(conversation), "turn": turn, "text": text}
            turn += 1


def nus_xml_rows(path: Path):
    root = ET.parse(path).getroot()
    count = 0
    for message in root.iter():
        if message.tag.rsplit("}", 1)[-1].lower() != "message":
            continue
        text_node = next((node for node in message if node.tag.rsplit("}", 1)[-1].lower() == "text"), None)
        text = (text_node.text or "").strip() if text_node is not None else ""
        if not text:
            continue
        user_node = next((node for node in message.iter()
                          if node.tag.rsplit("}", 1)[-1].lower() == "userid"), None)
        # Group by sender when metadata exists, preventing one person's recurring
        # SMS style and templates from appearing in both train and evaluation.
        group = (user_node.text or "").strip() if user_node is not None else ""
        group = group or message.attrib.get("id", str(count))
        yield {"conversation_id": group, "message_id": message.attrib.get("id"), "text": text}
        count += 1
    if count == 0:
        raise RuntimeError("no SMS text nodes found; inspect XML tags and extend the accepted set")


def _taskmaster_conversations(value):
    if isinstance(value, list):
        for item in value:
            yield from _taskmaster_conversations(item)
    elif isinstance(value, dict):
        utterances = value.get("utterances")
        if isinstance(utterances, list):
            yield value
        else:
            for child in value.values():
                if isinstance(child, (list, dict)):
                    yield from _taskmaster_conversations(child)


def taskmaster_rows(path: Path):
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    conversation_index = 0
    for json_path in paths:
        try:
            value = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for conversation in _taskmaster_conversations(value):
            group = str(conversation.get("conversation_id", conversation.get("id", conversation_index)))
            for turn, utterance in enumerate(conversation["utterances"]):
                text = utterance.get("text", utterance.get("utterance", ""))
                if isinstance(text, str) and text.strip():
                    yield {"conversation_id": group, "turn": turn, "text": text.strip()}
            conversation_index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("plain", "dailydialog", "dailydialog_jsonl",
                                              "gutenberg", "nus_xml", "taskmaster"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip conversion when the final atomic output already exists")
    args = parser.parse_args()
    if args.resume and args.output.exists():
        print(f"resume: already completed {args.output}")
        return
    readers = {"plain": plain_rows, "dailydialog": dailydialog_rows,
               "dailydialog_jsonl": dailydialog_jsonl_rows,
               "gutenberg": gutenberg_rows,
               "nus_xml": nus_xml_rows, "taskmaster": taskmaster_rows}
    count = write_jsonl(args.output, readers[args.format](args.input),
                        desc=f"convert {args.format}", atomic=True)
    if not count:
        raise RuntimeError(f"no rows converted from {args.input}")
    print(f"converted {count:,} utterances to {args.output}")


if __name__ == "__main__":
    main()
