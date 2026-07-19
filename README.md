# Qwen-distilled keyboard LM

This standalone English-first project does not write into the Xcode project. The
student has **10,601,472 parameters**, a 6,000-piece SentencePiece unigram
vocabulary with byte fallback, tied embeddings, six causal decoder blocks, and a
32-piece context. `<te>` and `<xlit>` are reserved so Telugu and Tenglish can be
added later without changing the runtime API. Qwen is an offline teacher and is
never included in the keyboard extension.

## Data mix

Stage A is 200M sampled tokens: OpenSubtitles English 45%, Taskmaster 15%,
Gutenberg Dialogue 15%, Leipzig English news sentences 15%, and DailyDialog
10%. Stage B is 20M sampled tokens: NUS SMS English 55%, DailyDialog 20%,
OpenSubtitles 15%, and Taskmaster 10%. Reddit is not used. Telugu and Tenglish are
not mixed into this English checkpoint.

## 1. Environment

```bash
cd /Users/meher/Downloads/qwen_lm_distil
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -v
```

Runtime selection is automatic: CUDA first, then MPS, then CPU. CUDA uses BF16 on
supported GPUs and FP16 otherwise; MPS defaults to FP32. If the 1.7B teacher is impractically slow, pass
`--model-id Qwen/Qwen3-0.6B-Base`; never change teachers halfway through one cache.

### Kaggle CUDA setup

Enable a GPU in Kaggle Notebook Settings, copy/upload this folder into the writable
working directory, then run:

```bash
# If this project was attached as a Kaggle Dataset, first copy its read-only folder.
cp -a /kaggle/input/YOUR_PROJECT_DATASET/qwen_lm_distil /kaggle/working/
cd /kaggle/working/qwen_lm_distil
python -m pip install -r requirements-kaggle.txt
python check_runtime.py --device cuda --precision auto
```

If Kaggle assigns a Tesla P100 and the runtime check says `sm_60` is missing,
replace Kaggle's CUDA 12.8 PyTorch wheel with the CUDA 12.6 build, which retains
Pascal support. The project does not need torchvision or torchaudio:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-cache-dir torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda'))"
python check_runtime.py --device cuda --precision fp16
```

`requirements-kaggle.txt` deliberately does not reinstall PyTorch, preserving
Kaggle's CUDA-enabled build. `/kaggle/input` is read-only; datasets, checkpoints,
and teacher shards must be written under `/kaggle/working`. This implementation is
single-GPU; a notebook exposing two T4 GPUs will use `cuda:0`, not both.

## 2. Convert and normalize downloads

Download every configured English source, or only selected sources. The
OpenSubtitles file is approximately 3.66GB compressed, so selective downloading is
useful while testing:

```bash
./download.sh all

# Examples of selective/resumable downloads:
./download.sh nus_sms dailydialog taskmaster
./download.sh opensubtitles gutenberg leipzig
```

`KLM_RAW_DIR=/another/path ./download.sh all` changes the raw-data location. The
script resumes partial downloads, validates archives, extracts them, and creates the
filenames below. If you downloaded the corpora yourself, put them under `data/raw`
and adjust the example filenames as necessary:

```bash
python convert_source.py --format taskmaster --input data/raw/taskmaster --output data/raw/taskmaster.jsonl --resume
python convert_source.py --format gutenberg --input data/raw/gutenberg_dialogue.txt --output data/raw/gutenberg_dialogue.jsonl --resume
python convert_source.py --format dailydialog_jsonl --input data/raw/dailydialog_extracted --output data/raw/dailydialog.jsonl --resume
python convert_source.py --format plain --input data/raw/leipzig.txt --output data/raw/leipzig.jsonl --resume
python convert_source.py --format nus_xml --input data/raw/nus_sms.xml --output data/raw/nus_sms.jsonl --resume

python prepare_data.py --source opensubtitles --input data/raw/opensubtitles.txt \
  --mode en --keep-fraction 0.04 --resume

for source in taskmaster gutenberg_dialogue dailydialog leipzig nus_sms; do
  python prepare_data.py --source "$source" --input "data/raw/$source.jsonl" --mode en --resume
done
```

The deterministic 4% OpenSubtitles sample is still roughly 17 million raw lines,
enough for its 90M-token Stage-A allocation. Split files are streamed directly to
disk. The downloader now fully extracts every archive and reports download and
extraction progress. Normalization, mixing, tokenizer sampling, teacher caching,
training, evaluation, quantization verification, and Core ML benchmarking also
show live progress bars.

Conversation IDs keep all turns from one dialogue in the same split. The English
cleaner preserves apostrophes: `we'll` remains different from `well`. Whole-word
scoring sums every SentencePiece piece, and the trie treats contractions as words.

## 3. Weighted corpora and tokenizer

```bash
python mix_stage.py --stage a --split train --resume
python mix_stage.py --stage b --split train --resume
python mix_stage.py --stage b --split dev --target-tokens 1000000 --resume

python train_tokenizer.py \
  --input data/processed/stage_a_train.jsonl data/processed/stage_b_train.jsonl \
  --input-weights 1 1 --output-prefix artifacts/spm --vocab-size 6000 --resume
```

Do not replace the tokenizer after training starts. Every checkpoint stores its
SHA-256; training and evaluation abort on a mismatch. Keep `spm.model`,
`spm.vocab`, and `tokenizer_metadata.json` together. The 1:1 tokenizer sample is
intentional: it gives short-message spellings more vocabulary influence than their
raw corpus size while the LM itself still trains on the Stage A/B schedules above.

## 4. Stage A pretraining

```bash
python train.py --stage pretrain \
  --input data/processed/stage_a_train.jsonl \
  --output artifacts/pretrain.pt --device cuda --precision auto \
  --micro-batch 128 --accumulation 4 --checkpoint-every 250 --auto-resume
```

`--auto-resume` restores the output checkpoint, optimizer, scheduler position,
token count, and random state after interruption.

## 5. Qwen soft-target cache

Benchmark 100 examples before the long CUDA job:

```bash
python cache_teacher.py --input data/processed/stage_a_train.jsonl \
  --output data/processed/teacher_probe.jsonl.gz --max-examples 100 \
  --device cuda --teacher-batch-size 8 --score-micro-batch 32
```

Create restartable atomic shards and merge them. Rerunning the same command skips
completed shards and restarts only an interrupted shard:

```bash
python cache_teacher_shards.py --input data/processed/stage_a_train.jsonl \
  --output-dir data/teacher --prefix teacher_a --total-examples 100000 --shards 8 \
  --merge-output data/teacher/teacher_a_100k.jsonl.gz \
  --device cuda --teacher-batch-size 4 --score-micro-batch 16
```

Each row stores word candidates and teacher log-probabilities, not full teacher
logits. The shortlist is the union of trie typo corrections, context bigrams, Qwen
generation, and forced gold. Qwen scores a complete word using its tokenizer; the
student aligns at word level, so incompatible tokenizer IDs are never equated.

## Interactive keyboard probes

These use the Xcode project's real `langdata.bin`, weighted typo trie, n-gram
counts, blend weights, and correction gates while scoring with this project's
10.6M checkpoint. Before distillation they default to `artifacts/pretrain.pt`:

```bash
python next_words_probe.py "i am going" --device cpu
python correction_probe.py teh --context "i went to" --device cpu
python next_words_probe.py                         # interactive
python correction_probe.py                        # interactive
```

After distillation, select the new checkpoint without changing the harness:

```bash
python next_words_probe.py "i am going" --checkpoint artifacts/distill.pt --device cpu
python correction_probe.py teh --context "i went to" \
  --checkpoint artifacts/distill.pt --device cpu
```

The default legacy-tools location is
`/Users/meher/Downloads/xcode-projects/keyboard/tools/lm`. Override it with
`--legacy-tools /path/to/tools/lm` or the `KEYBOARD_LM_TOOLS` environment variable.

## 6. Distill, SMS-adapt, then QAT

Cross-stage `--resume` loads weights but starts a fresh optimizer and step counter.

```bash
python train.py --stage distill --input data/teacher/teacher_a_100k.jsonl.gz \
  --output artifacts/distill.pt --resume artifacts/pretrain.pt \
  --device cuda --precision auto --micro-batch 128 --accumulation 4 \
  --checkpoint-every 250 --auto-resume

python train.py --stage finetune --input data/processed/stage_b_train.jsonl \
  --output artifacts/finetune.pt --resume artifacts/distill.pt \
  --device cuda --precision auto --micro-batch 128 --accumulation 4 \
  --checkpoint-every 250 --auto-resume

python cache_teacher.py --input data/processed/stage_b_train.jsonl \
  --index-corpus data/processed/stage_b_train.jsonl \
  --output data/processed/teacher_b.jsonl.gz --max-examples 100000 \
  --device cuda --teacher-batch-size 8 --score-micro-batch 32 --resume

python train.py --stage qat --input data/processed/teacher_b.jsonl.gz \
  --output artifacts/qat.pt --resume artifacts/finetune.pt \
  --device cuda --precision auto --micro-batch 128 --accumulation 4 \
  --checkpoint-every 250 --auto-resume
```

The loss is 65% hard next-piece cross-entropy, 30% temperature-2 soft candidate
KL, and 5% periodic whole-word sequence KL. QAT fake-quantizes embeddings and
linear weights per output channel while retaining FP32 master weights.

### Later Telugu and Tenglish run

The candidate pipeline recognizes Telugu Unicode words and the configuration uses
`Qwen/Qwen3-4B-Base` for `<te>` and `<xlit>` teacher caches. Pass `--mode te` or
`--mode xlit` to `cache_teacher_shards.py`. On a 16GB P100, start with
`--teacher-batch-size 1 --score-micro-batch 8`. A production multilingual model
must train a new shared SentencePiece model on English, Telugu, and Tenglish before
pretraining; byte fallback alone is a compatibility fallback, not adequate Telugu
segmentation. Changing the tokenizer after student training invalidates its
embedding and output matrices.

## 7. Held-out evaluation

```bash
python cache_teacher.py --input data/processed/stage_b_dev.jsonl \
  --index-corpus data/processed/stage_b_train.jsonl \
  --output data/processed/teacher_b_dev.jsonl.gz --max-examples 10000 \
  --device cuda --teacher-batch-size 8 --score-micro-batch 32 --resume

python evaluate.py --checkpoint artifacts/qat.pt \
  --input data/processed/teacher_b_dev.jsonl.gz --output artifacts/eval.json --device cuda

python verify_quantization.py --checkpoint artifacts/qat.pt \
  --input data/processed/teacher_b_dev.jsonl.gz --device cuda
```

`candidate_recall` compares trie, ngram, Qwen, and their union; forced-gold
insertion is excluded. `conditional_ranking` measures student top-1, top-3, and
MRR when the gold is in the cached union. This separates candidate-generation
failure from neural-ranking failure.

The int8 gate permits at most 1 percentage point absolute top-3 loss and 1%
relative MRR loss. It also requires 99% top-1 agreement and 0.995 score
correlation. A failure means continue QAT or inspect unstable layers.

## 8. Core ML export

Core ML conversion and prediction require macOS. After CUDA training, download
`artifacts/qat.pt`, `artifacts/spm.model`, and
`artifacts/tokenizer_metadata.json` from Kaggle into the same relative paths in
this project on your Mac, then run:

```bash
python export_coreml.py --checkpoint artifacts/qat.pt --graph next --smoke-predict
python benchmark_coreml.py --model artifacts/coreml/KeyboardLMNext_int8.mlpackage --graph next

# Optional ablation; do not ship both packages.
python export_coreml.py --checkpoint artifacts/qat.pt --graph candidate
python benchmark_coreml.py --model artifacts/coreml/KeyboardLMScorer_int8.mlpackage --graph candidate
```

The exporter intentionally does not import SentencePiece, avoiding the old
protobuf/SentencePiece mutex hang. The default single-context graph returns the
6,000 next-piece logits used by trie-constrained beam search. The optional scorer
uses a fixed 16-candidate batch for the latency/quality ablation. Both packages
contain the same weights, so shipping both would waste the budget. The chosen int8
package must be at most 12MB, leaving roughly 2MB for tokenizer and integration
overhead. Nothing is copied into Xcode automatically.

## Ship gates and likely difficulties

- No conversation leakage or tokenizer hash mismatch.
- Candidate union recall beats each individual generator.
- SMS held-out top-3 and MRR improve after Stage B.
- Simulated int8 stays inside the 1% quality gates.
- Actual Core ML int8 is at most 12MB and passes graph prediction.
- Measure on-device p50/p95 latency, peak extension memory, cold start, and
  suggestion acceptance; offline top-3 alone is not a ship criterion.

Qwen cache generation remains the longest CUDA stage, subtitle language differs
from SMS, repeated templates can leak if group IDs are lost, proper names remain
difficult, and CUDA batch sizes may need reducing on a T4. Sharding handles
interruption risk; source-level metrics and candidate ablation expose
domain/candidate problems; QAT addresses quantization loss. The size is feasible,
but success depends on held-out SMS quality and on-device latency rather than
parameter count alone.
