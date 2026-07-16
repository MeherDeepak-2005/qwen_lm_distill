#!/usr/bin/env bash
# Download and extract the English corpora expected by convert_source.py.
# Usage: ./download_sources.sh [all|nus_sms|dailydialog|opensubtitles|taskmaster|gutenberg|leipzig ...]

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${KLM_RAW_DIR:-$ROOT/data/raw}"
mkdir -p "$RAW"

VALID=" all nus_sms dailydialog opensubtitles taskmaster gutenberg leipzig "
SELECTED=("$@")
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=(all)
fi
for source in "${SELECTED[@]}"; do
  if [[ "$VALID" != *" $source "* ]]; then
    echo "Unknown source: $source" >&2
    echo "Choose from: all nus_sms dailydialog opensubtitles taskmaster gutenberg leipzig" >&2
    exit 2
  fi
done

for command in curl unzip gzip tar find awk cut; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done

want() {
  local requested
  for requested in "${SELECTED[@]}"; do
    [[ "$requested" == all || "$requested" == "$1" ]] && return 0
  done
  return 1
}

fetch() {
  local url="$1"
  local output="$2"
  local partial="$output.part"
  if [[ -s "$output" ]]; then
    echo "Already downloaded: $output"
    return
  fi
  echo "Downloading: $url"
  curl --fail --location --retry 5 --retry-delay 2 --connect-timeout 30 \
       --continue-at - --output "$partial" "$url"
  [[ -s "$partial" ]] || { echo "Download produced an empty file: $partial" >&2; exit 1; }
  mv "$partial" "$output"
}

if want nus_sms; then
  archive="$RAW/nus_sms_en_xml.zip"
  fetch "https://raw.githubusercontent.com/WING-NUS/nus-sms-corpus/master/smsCorpus_en_xml_2015.03.09_all.zip" "$archive"
  unzip -tq "$archive" >/dev/null
  if [[ ! -s "$RAW/nus_sms.xml" ]]; then
    unzip -p "$archive" 'smsCorpus_en_2015.03.09_all.xml' > "$RAW/nus_sms.xml.tmp"
    mv "$RAW/nus_sms.xml.tmp" "$RAW/nus_sms.xml"
  fi
fi

if want dailydialog; then
  archive="$RAW/dailydialog_raw.zip"
  directory="$RAW/dailydialog_extracted"
  fetch "https://linqs-data.soe.ucsc.edu/public/datasets/dailydialog/dailydialog-raw.zip" "$archive"
  unzip -tq "$archive" >/dev/null
  mkdir -p "$directory"
  unzip -oq "$archive" -d "$directory"
  find "$directory" -type f -name 'train.json' -print -quit | grep -q . || {
    echo "train.json missing from DailyDialog archive" >&2
    exit 1
  }
fi

if want opensubtitles; then
  archive="$RAW/opensubtitles_en.txt.gz"
  echo "OpenSubtitles is approximately 3.66 GB compressed and needs substantial extraction space."
  fetch "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz" "$archive"
  gzip -t "$archive"
  if [[ ! -s "$RAW/opensubtitles.txt" ]]; then
    gzip -dc "$archive" > "$RAW/opensubtitles.txt.tmp"
    mv "$RAW/opensubtitles.txt.tmp" "$RAW/opensubtitles.txt"
  fi
fi

if want taskmaster; then
  archive="$RAW/taskmaster.zip"
  directory="$RAW/taskmaster"
  fetch "https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip" "$archive"
  unzip -tq "$archive" >/dev/null
  mkdir -p "$directory"
  unzip -oq "$archive" -d "$directory"
  find "$directory" -type f -name '*.json' -print -quit | grep -q . || {
    echo "No JSON files found in extracted Taskmaster archive" >&2
    exit 1
  }
fi

if want gutenberg; then
  directory="$RAW/gutenberg_download"
  mkdir -p "$directory"
  # Plain-text mirror of the published English Gutenberg Dialogue files.
  base="https://huggingface.co/datasets/willwade/Gutenberg-dialog-en/resolve/main"
  for split in train dev test; do
    fetch "$base/$split.txt?download=true" "$directory/$split.txt"
  done
  if [[ ! -s "$RAW/gutenberg_dialogue.txt" ]]; then
    {
      cat "$directory/train.txt"
      printf '\n'
      cat "$directory/dev.txt"
      printf '\n'
      cat "$directory/test.txt"
      printf '\n'
    } > "$RAW/gutenberg_dialogue.txt.tmp"
    mv "$RAW/gutenberg_dialogue.txt.tmp" "$RAW/gutenberg_dialogue.txt"
  fi
fi

if want leipzig; then
  archive="$RAW/eng_news_2024_1M.tar.gz"
  directory="$RAW/leipzig_extracted"
  fetch "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2024_1M.tar.gz" "$archive"
  tar -tzf "$archive" >/dev/null
  mkdir -p "$directory"
  tar -xzf "$archive" -C "$directory"
  sentence_file="$(find "$directory" -type f -name '*-sentences.txt' | sed -n '1p')"
  [[ -n "$sentence_file" ]] || { echo "Leipzig sentences file missing from archive" >&2; exit 1; }
  cut -f2- "$sentence_file" > "$RAW/leipzig.txt.tmp"
  mv "$RAW/leipzig.txt.tmp" "$RAW/leipzig.txt"
fi

echo
echo "Downloads prepared under: $RAW"
for output in nus_sms.xml dailydialog_extracted opensubtitles.txt taskmaster gutenberg_dialogue.txt leipzig.txt; do
  [[ -e "$RAW/$output" ]] && du -sh "$RAW/$output"
done
echo "Next: follow README.md section 'Convert and normalize downloads'."
