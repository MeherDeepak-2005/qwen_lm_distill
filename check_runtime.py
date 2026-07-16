#!/usr/bin/env python3
"""Fail-fast accelerator check for Kaggle and local machines."""

from __future__ import annotations

import argparse
import json

import torch

from runtime import (DEVICE_CHOICES, PRECISION_CHOICES, configure_device,
                     device_report, resolve_device, resolve_precision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    precision, amp_dtype = resolve_precision(args.precision, device)
    configure_device(device)
    left = torch.randn((1024, 1024), device=device)
    right = torch.randn((1024, 1024), device=device)
    if amp_dtype is None:
        result = left @ right
    else:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            result = left @ right
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = device_report(device, precision)
    report.update({"matmul_ok": bool(torch.isfinite(result).all().item()),
                   "result_dtype": str(result.dtype)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
