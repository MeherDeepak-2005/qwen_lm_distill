#!/usr/bin/env python3
"""Export one compact variable-batch Core ML graph and per-channel int8 weights.

This file intentionally does not import SentencePiece. Keeping tokenizer/protobuf
loading out of the Core ML process avoids the mutex hang seen in older exporters.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from model import (CompactInferenceModel, KeyboardDecoder, ModelConfig,
                   NextTokenInferenceModel, count_parameters)
from utils import ROOT, load_config


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_student(checkpoint_path: Path) -> KeyboardDecoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = ModelConfig.from_dict(checkpoint["model_config"])
    model = KeyboardDecoder(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval().enable_qat(False)
    print(f"loaded {count_parameters(model):,} parameters")
    return model


def smoke_predict(fp16_model, int8_model, vocab_size: int, max_length: int, graph: str) -> dict:
    agreements, correlations = [], []
    rng = np.random.default_rng(17)
    for _ in range(8):
        length = int(rng.integers(4, max_length + 1))
        batch = 1 if graph == "next" else 16
        tokens = np.full((batch, max_length), 3, dtype=np.int32)
        tokens[:, :length] = rng.integers(4, vocab_size, size=(batch, length), dtype=np.int32)
        inputs = {"tokens": tokens, "next_positions": np.asarray([[length - 1]], dtype=np.int32)}
        if graph == "candidate":
            inputs.update({
                "score_positions": np.tile(np.arange(8, dtype=np.int32), (16, 1)),
                "target_ids": rng.integers(4, vocab_size, size=(16, 8), dtype=np.int32),
                "score_mask": np.ones((16, 8), dtype=np.float32),
            })
        fp = fp16_model.predict(inputs)["next_logits"].reshape(-1).astype(np.float64)
        quant = int8_model.predict(inputs)["next_logits"].reshape(-1).astype(np.float64)
        agreements.append(int(fp.argmax() == quant.argmax()))
        correlation = np.corrcoef(fp, quant)[0, 1]
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
    return {"contexts": 8, "top1_agreement": float(np.mean(agreements)),
            "minimum_logit_correlation": min(correlations),
            "mean_logit_correlation": float(np.mean(correlations))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "coreml")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--graph", choices=("next", "candidate"), default="next")
    parser.add_argument("--smoke-predict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    basename = "KeyboardLMNext" if args.graph == "next" else "KeyboardLMScorer"
    expected_manifest = args.output_dir / f"{basename}_manifest.json"
    expected_int8 = args.output_dir / f"{basename}_int8.mlpackage"
    if args.resume and expected_manifest.exists() and expected_int8.exists():
        print(f"resume: Core ML export already completed: {expected_int8}")
        return

    try:
        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
        )
    except ImportError as error:
        raise SystemExit("Install coremltools>=9 in this environment before export") from error

    student = load_student(args.checkpoint)
    cfg = student.config
    if args.graph == "next":
        wrapper = NextTokenInferenceModel(student).eval()
        examples = (torch.zeros((1, cfg.max_length), dtype=torch.int32),
                    torch.zeros((1, 1), dtype=torch.int32))
        coreml_inputs = [
            ct.TensorType(name="tokens", shape=(1, cfg.max_length), dtype=np.int32),
            ct.TensorType(name="next_positions", shape=(1, 1), dtype=np.int32),
        ]
        coreml_outputs = [ct.TensorType(name="next_logits")]
        basename = "KeyboardLMNext"
    else:
        wrapper = CompactInferenceModel(student).eval()
        batch = 16
        examples = (
            torch.zeros((batch, cfg.max_length), dtype=torch.int32),
            torch.zeros((batch, wrapper.score_slots), dtype=torch.int32),
            torch.zeros((batch, wrapper.score_slots), dtype=torch.int32),
            torch.ones((batch, wrapper.score_slots), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.int32),
        )
        coreml_inputs = [
            ct.TensorType(name="tokens", shape=(batch, cfg.max_length), dtype=np.int32),
            ct.TensorType(name="score_positions", shape=(batch, wrapper.score_slots), dtype=np.int32),
            ct.TensorType(name="target_ids", shape=(batch, wrapper.score_slots), dtype=np.int32),
            ct.TensorType(name="score_mask", shape=(batch, wrapper.score_slots), dtype=np.float32),
            ct.TensorType(name="next_positions", shape=(1, 1), dtype=np.int32),
        ]
        coreml_outputs = [ct.TensorType(name="candidate_log_probs"), ct.TensorType(name="next_logits")]
        basename = "KeyboardLMScorer"
    print("tracing PyTorch inference graph...", flush=True)
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(wrapper, examples, strict=True))

    print("converting traced graph to Core ML FP16...", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        inputs=coreml_inputs,
        outputs=coreml_outputs,
    )
    mlmodel.author = "qwen_lm_distil"
    mlmodel.short_description = "English keyboard next-word decoder with trie candidate scoring"
    mlmodel.user_defined_metadata["vocab_size"] = str(cfg.vocab_size)
    mlmodel.user_defined_metadata["max_length"] = str(cfg.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = args.output_dir / f"{basename}_fp16.mlpackage"
    int8_path = args.output_dir / f"{basename}_int8.mlpackage"
    for path in (fp16_path, int8_path):
        if path.exists():
            shutil.rmtree(path)
    print(f"saving FP16 package: {fp16_path}", flush=True)
    mlmodel.save(fp16_path)
    quant_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=1024,
    ))
    print("quantizing Core ML weights to per-channel int8...", flush=True)
    int8_model = linear_quantize_weights(mlmodel, config=quant_config)
    print(f"saving int8 package: {int8_path}", flush=True)
    int8_model.save(int8_path)

    settings = load_config(args.config)["quantization"]
    sizes = {"fp16_mb": package_bytes(fp16_path) / 1_000_000,
             "int8_mb": package_bytes(int8_path) / 1_000_000}
    result = {"checkpoint": str(args.checkpoint), "graph": args.graph,
              "parameters": count_parameters(student),
              "packages": {"fp16": str(fp16_path), "int8": str(int8_path)}, "sizes": sizes,
              "size_gate_mb": settings["max_package_mb"],
              "size_gate_passed": sizes["int8_mb"] <= settings["max_package_mb"]}
    if args.smoke_predict:
        result["coreml_smoke"] = smoke_predict(
            mlmodel, int8_model, cfg.vocab_size, cfg.max_length, args.graph,
        )
    manifest = args.output_dir / f"{basename}_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["size_gate_passed"]:
        raise SystemExit(f"int8 package exceeds {settings['max_package_mb']} MB gate")


if __name__ == "__main__":
    main()
