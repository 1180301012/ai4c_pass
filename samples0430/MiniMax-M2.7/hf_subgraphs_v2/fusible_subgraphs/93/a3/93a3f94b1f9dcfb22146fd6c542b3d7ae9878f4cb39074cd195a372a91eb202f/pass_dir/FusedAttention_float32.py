import torch
import triton
import triton.language as tl

# Import the shared kernel functions from FusedAttention.py
from pass_dir.FusedAttention import (
    fused_attention_autotuned,
    triton_fused_attention_float16,
    triton_fused_attention_bfloat16,
    triton_fused_attention_float32,
)


def pattern(in_0, in_1, in_2):
    """
    Match the core attention computation pattern:
    matmul -> scale -> softmax -> matmul
    Float32 variant.
    """
    # First matmul: Q @ K^T
    tmp_0 = torch.matmul(in_0, in_1)
    # Scale by 1.0 (no-op but part of the pattern)
    tmp_1 = tmp_0 * 1.0
    # Softmax
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    # Type conversion (no-op for float32)
    tmp_3 = tmp_2.to(torch.float32)
    # Dropout with p=0.0 (no-op)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    # Second matmul: attention @ V
    tmp_5 = torch.matmul(tmp_4, in_2)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "float32")


# Module-level dispatch wrapper function
def fused_attention_dispatch(q, k, v, route):
    if route == "bfloat16":
        output = triton_fused_attention_bfloat16(q, k, v)
    elif route == "float16":
        output = triton_fused_attention_float16(q, k, v)
    elif route == "float32":
        output = triton_fused_attention_float32(q, k, v)
    else:
        raise ValueError(f"Unknown route: {route}")
    
    return output


def replacement_func():
    return fused_attention_dispatch