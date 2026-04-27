import torch
from pass_dir.kernels import shared_dispatch


# ── pattern ──────────────────────────────────────────────────────────────────
# SmolLM3: cos(cat_result) → ×1.0 → cast bfloat16  (takes cat result as input)
def pattern(cat_result):
    cos_val  = cat_result.cos()
    cos_sc   = cos_val * 1.0
    cos_cast = cos_sc.to(dtype=torch.bfloat16)
    return cos_cast


def replacement_args(cat_result):
    return (cat_result, "cos_bf16")


def replacement_func():
    return shared_dispatch