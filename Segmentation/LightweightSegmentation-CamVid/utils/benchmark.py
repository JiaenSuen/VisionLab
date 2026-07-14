from __future__ import annotations

import time

import torch


@torch.inference_mode()
def benchmark_model(
    model,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup: int = 50,
    iterations: int = 200,
    use_fp16: bool = False,
) -> dict:
    model.eval().to(device)
    sample = torch.randn(input_shape, device=device)
    enabled = use_fp16 and device.type == "cuda"

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, enabled=enabled, dtype=torch.float16):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.autocast(device_type=device.type, enabled=enabled, dtype=torch.float16):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000 / iterations
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": parameters,
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms,
        "peak_memory_mb": peak_memory_mb,
        "precision": "fp16" if enabled else "fp32",
        "input_shape": list(input_shape),
    }

