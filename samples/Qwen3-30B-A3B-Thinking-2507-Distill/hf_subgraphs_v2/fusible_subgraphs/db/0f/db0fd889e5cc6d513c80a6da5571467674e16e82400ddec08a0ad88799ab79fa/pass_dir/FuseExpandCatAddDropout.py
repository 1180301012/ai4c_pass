"""
Fuse: add + dropout(training=False)

Pattern: takes two tensors x, y, adds them, applies dropout with p=0.1 and
training=False. Since training=False dropout is an identity, we only need to
compute x + y.

This pattern appears twice in the YOLOS model:
  tmp_23 = tmp_12 + tmp_22
  tmp_24 = dropout(tmp_23, 0.1, False, False)

  tmp_25 = in_6[:, :, 0, :]
  tmp_26 = tmp_25[:, None]
  ... (path 2 operations)

We fuse add + dropout into a single Triton kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: element-wise add  (dropout with training=False is identity)
# ---------------------------------------------------------------------------

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_dropout(x, y):
    """
    Replaces dropout(x + y, 0.1, training=False) with just x + y.
    """
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(x, y):
    """Matches: result = x + y followed by dropout(result, 0.1, False, False)."""
    tmp_23 = x + y
    tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
    return tmp_24


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return fused_add_dropout