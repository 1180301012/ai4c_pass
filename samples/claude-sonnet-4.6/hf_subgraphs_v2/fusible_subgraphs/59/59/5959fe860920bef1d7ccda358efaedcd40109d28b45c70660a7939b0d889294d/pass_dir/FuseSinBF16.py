import torch
from pass_dir.kernels import shared_dispatch


# ── pattern ──────────────────────────────────────────────────────────────────
# SmolLM3: sin(cat_result) → ×1.0 → cast bfloat16  (takes cat result as input)
def pattern(cat_result):
    sin_val  = cat_result.sin()
    sin_sc   = sin_val * 1.0
    sin_cast = sin_sc.to(dtype=torch.bfloat16)
    return sin_cast


def replacement_args(cat_result):
    return (cat_result, "sin_bf16")


def replacement_func():
    return shared_dispatch