"""GPU-dependent training settings. The effective batch stays 8 on every GPU, so lr, LoRA and data stay comparable
between an A100 run and a T4 run; only how the 8 rows are split into steps changes."""

EFFECTIVE_BATCH = 8


def batch_shape(gpu_name: str) -> tuple[int, int]:
    """(per-device batch, grad-accum) for a `torch.cuda.get_device_name` value. An A100 (40 or 80 GB) holds all 8
    rows of a 4-bit 9B model at 2048 tokens in one step; any other GPU (T4) keeps the measured 2 x 4."""
    return (8, 1) if "A100" in gpu_name.upper() else (2, 4)


def precision_flags(compute_dtype: str) -> dict[str, bool]:
    """SFTConfig mixed-precision flags for the model's compute dtype (`str(torch dtype)`): bf16 autocast for
    bfloat16 (A100), fp16 for float16 (T4), neither for float32."""
    return {"fp16": compute_dtype == "torch.float16", "bf16": compute_dtype == "torch.bfloat16"}
