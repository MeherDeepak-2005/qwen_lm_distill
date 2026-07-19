#!/usr/bin/env python3
"""CUDA/MPS pretraining, word-level soft distillation, SMS fine-tuning and QAT."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets import packed_batches, record_batches
from losses import distillation_loss
from model import KeyboardDecoder, ModelConfig, count_parameters
from runtime import (DEVICE_CHOICES, PRECISION_CHOICES, configure_device,
                     peak_memory_mb, print_device_report, resolve_device,
                     resolve_precision)
from utils import ROOT, cosine_lr, load_config, seed_everything, tokenizer_hash


def save_checkpoint(path: Path, model, optimizer, step: int, tokens: int, stage: str,
                    config: dict, tokenizer_path: Path, best_loss: float) -> None:
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "tokens": tokens, "stage": stage, "config": config,
        "model_config": model.checkpoint_metadata()["model_config"],
        "tokenizer_sha256": tokenizer_hash(tokenizer_path), "best_loss": best_loss,
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pretrain", "distill", "finetune", "qat"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "spm.model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "student.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--auto-resume", action="store_true",
                        help="Resume --output when present; otherwise initialize from --resume")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=DEVICE_CHOICES)
    parser.add_argument("--precision", choices=PRECISION_CHOICES)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--steps", type=int, help="Distillation/QAT step override")
    parser.add_argument("--micro-batch", type=int, help="Override sequences per forward pass")
    parser.add_argument("--accumulation", type=int, help="Override gradient accumulation")
    parser.add_argument("--checkpoint-every", type=int,
                        help="Override checkpoint interval; the same atomic file is replaced")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    train_cfg, model_cfg = cfg["training"], ModelConfig.from_dict(cfg["model"])
    seed_everything(train_cfg["seed"])
    device = resolve_device(args.device or train_cfg["device"])
    precision, amp_dtype = resolve_precision(args.precision or train_cfg["precision"], device)
    configure_device(device)
    print_device_report(device, precision)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tokenizer.vocab_size() != model_cfg.vocab_size:
        raise ValueError(f"tokenizer={tokenizer.vocab_size()} model={model_cfg.vocab_size}")
    model = KeyboardDecoder(model_cfg).to(device)

    lr = {
        "pretrain": train_cfg["pretrain_lr"], "distill": train_cfg["distill_lr"],
        "finetune": train_cfg["finetune_lr"], "qat": train_cfg["finetune_lr"] * 0.5,
    }[args.stage]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg["weight_decay"])
    start_step = tokens_seen = 0
    best_loss = float("inf")
    resume_path = args.output if args.auto_resume and args.output.exists() else args.resume
    if resume_path:
        print(json.dumps({"loading_checkpoint": str(resume_path),
                          "auto_resume": bool(args.auto_resume and resume_path == args.output)}),
              flush=True)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("tokenizer_sha256") != tokenizer_hash(args.tokenizer):
            raise RuntimeError("checkpoint/tokenizer hash mismatch")
        same_stage = args.stage == checkpoint.get("stage")
        if checkpoint.get("optimizer") and same_stage:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint.get("step", 0)
            tokens_seen = checkpoint.get("tokens", 0)
            best_loss = checkpoint.get("best_loss", best_loss)
            if checkpoint.get("torch_rng") is not None:
                torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        else:
            print(json.dumps({"loaded_weights_from_stage": checkpoint.get("stage"),
                              "starting_stage": args.stage, "step": 0}), flush=True)
    if args.stage == "qat":
        model.enable_qat(True)

    batch_size = args.micro_batch or train_cfg["micro_batch_sequences"]
    accumulation = args.accumulation or train_cfg["gradient_accumulation"]
    checkpoint_every = args.checkpoint_every or train_cfg["checkpoint_every"]
    if batch_size < 1 or accumulation < 1 or checkpoint_every < 1:
        raise ValueError("batch size and accumulation must be positive")
    if args.stage in ("pretrain", "finetune"):
        default_target = cfg["data"]["stage_a_target_tokens" if args.stage == "pretrain" else "stage_b_target_tokens"]
        target_tokens = args.target_tokens or default_target
        total_steps = math.ceil(target_tokens / (batch_size * model_cfg.max_length * accumulation))
        batches = packed_batches(args.input, tokenizer, batch_size, model_cfg.max_length,
                                 train_cfg["seed"] + (0 if args.stage == "pretrain" else 2))
    else:
        total_steps = args.steps or (20_000 if args.stage == "distill" else 2_000)
        batches = record_batches(args.input, max(2, batch_size // 4), train_cfg["seed"] + 3)

    use_amp = amp_dtype is not None
    use_scaler = device.type == "cuda" and precision == "fp16"
    if hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=use_scaler)
    else:
        # PyTorch 2.0 compatibility. This is a no-op under the default FP32
        # configuration; production requirements pin the supported 2.7 line.
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    progress = tqdm(range(start_step, total_steps), total=total_steps, initial=start_step,
                    desc=f"train {args.stage}", unit="step", dynamic_ncols=True,
                    mininterval=0.5)
    for step in progress:
        learning_rate = cosine_lr(step, total_steps, train_cfg["warmup_steps"], lr)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        running = {"loss": 0.0, "hard": 0.0, "soft": 0.0, "sequence": 0.0}
        for micro in range(accumulation):
            batch = next(batches)
            amp_context = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                           if use_amp else nullcontext())
            with amp_context:
                if args.stage in ("pretrain", "finetune"):
                    batch = batch.to(device, non_blocking=device.type == "cuda")
                    logits = model(batch[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
                    parts = {"hard": loss.detach(), "soft": torch.tensor(0), "sequence": torch.tensor(0)}
                    tokens_seen += batch_size * model_cfg.max_length
                else:
                    include_sequence = step % train_cfg["sequence_kd_every"] == 0 and micro == 0
                    loss, parts = distillation_loss(
                        model, batch, tokenizer, model_cfg.max_length, device,
                        cfg["teacher"]["temperature"], train_cfg["hard_loss_weight"],
                        train_cfg["soft_loss_weight"], train_cfg["sequence_loss_weight"],
                        include_sequence, train_cfg["sequence_kd_candidates"],
                    )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            running["loss"] += loss.detach().float().item() / accumulation
            for name in ("hard", "soft", "sequence"):
                running[name] += parts[name].float().item() / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if running["loss"] < best_loss:
            best_loss = running["loss"]
        progress.set_postfix(loss=f"{running['loss']:.4f}", lr=f"{learning_rate:.2e}",
                             tokens=f"{tokens_seen / 1_000_000:.1f}M", refresh=False)
        if (step + 1) % 20 == 0 or step == start_step:
            elapsed = max(time.time() - started, 1e-6)
            tqdm.write(json.dumps({"stage": args.stage, "step": step + 1, "steps": total_steps,
                                   "micro_batch": batch_size, "accumulation": accumulation,
                                   "tokens": tokens_seen, "lr": learning_rate, **running,
                                   "steps_per_second": (step + 1 - start_step) / elapsed,
                                   "peak_device_mb": peak_memory_mb(device)}))
        if (step + 1) % checkpoint_every == 0:
            tqdm.write(f"saving checkpoint: {args.output}")
            save_checkpoint(args.output, model, optimizer, step + 1, tokens_seen,
                            args.stage, cfg, args.tokenizer, best_loss)
    progress.close()
    save_checkpoint(args.output, model, optimizer, total_steps, tokens_seen,
                    args.stage, cfg, args.tokenizer, best_loss)
    print(f"saved {args.output}; {count_parameters(model):,} parameters on {device}")


if __name__ == "__main__":
    main()
