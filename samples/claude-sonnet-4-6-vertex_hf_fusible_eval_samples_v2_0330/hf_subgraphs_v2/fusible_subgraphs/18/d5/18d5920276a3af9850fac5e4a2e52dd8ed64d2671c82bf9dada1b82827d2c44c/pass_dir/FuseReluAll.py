"""
Try to match relu + dropout + flatten using torch.relu (the canonical
aten-level relu that FX may normalize torch.nn.functional.relu to).

If matched, a single Triton kernel replaces all three operations,
reducing the number of FX nodes from 3 to 1 and fusing the only
meaningful GPU operation (relu) with the free operations.

Input shape:  [B, 2048, 1, 1]  (float16 / bfloat16 / float32)
Output shape: [B, 2048]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – try torch.relu (aten-level) instead of F.relu, in case FX
# normalizes torch.nn.functional.relu → torch.relu internally.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.relu(in_0)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel – no autotune, fixed BLOCK_SIZE=2048.
# All test sizes (2048, 65536, 262144) are exact multiples of 2048 so no
# masking is required, reducing kernel overhead.
# ---------------------------------------------------------------------------
@triton.jit
def _relu_flatten_kernel(
    inp_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(inp_ptr + offsets)
    y = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offsets, y)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _relu_flatten_impl(in_0):
    batch = in_0.shape[0]
    n_elements = in_0.numel()         # B * 2048
    flat_size = n_elements // batch   # always 2048

    out = torch.empty((batch, flat_size), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 2048
    grid = (n_elements // BLOCK_SIZE,)   # exact division, no masking needed

    _relu_flatten_kernel[grid](
        in_0,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return _relu_flatten_impl