"""Portable CPU, Apple MPS and NVIDIA CUDA runtime configuration."""

from __future__ import annotations

import json

import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif mps_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested but unavailable (torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda}). Enable a Kaggle GPU accelerator."
        )
    if requested == "mps" and not mps_available():
        raise RuntimeError("MPS requested but unavailable; use --device cpu or --device cuda")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    if requested == "auto":
        if device.type == "cuda":
            requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            requested = "fp32"
    if requested == "fp32":
        return requested, None
    if device.type not in ("cuda", "mps"):
        raise RuntimeError(f"{requested} autocast is unsupported by this pipeline on {device.type}")
    if requested == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this GPU does not report native bf16 support")
        return requested, torch.bfloat16
    return requested, torch.float16


def configure_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def device_report(device: torch.device, precision: str | None = None) -> dict:
    report = {"device": str(device), "torch": torch.__version__, "precision": precision}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report.update({
            "gpu": properties.name,
            "cuda_build": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(index)),
            "vram_gb": round(properties.total_memory / 1_000_000_000, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    elif device.type == "mps":
        report["gpu"] = "Apple Metal Performance Shaders"
    return report


def print_device_report(device: torch.device, precision: str | None = None) -> None:
    print(json.dumps({"runtime": device_report(device, precision)}), flush=True)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1_000_000
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1_000_000
    return None
