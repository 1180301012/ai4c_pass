"""
Optimized replacement for dropout(p=0.0, training=False) + flatten(1,-1).

Since dropout with p=0.0 and training=False is a pure identity (no-op),
and flatten on a contiguous [B, C, 1, 1] tensor is just a view to [B, C],
both operations can be replaced with a single Python reshape — no CUDA
kernel needed.  This eliminates:
  - the C++ overhead of the dropout dispatch
  - the flatten dispatch
and replaces both with one lightweight view call.

Input shape to pattern:  relu's output [B, 2048, 1, 1]
Output shape from pattern: [B, 2048]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: the two identity/view operations that follow relu
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel – element-wise ReLU applied to the input; output laid out
# as [B, C*H*W].  This is used when pattern input is BEFORE relu.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu_flatten_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper: dropout(identity) + flatten replaced by a pure view.
# No GPU kernel is launched – the tensor is already relu-ed (all ≥ 0) and
# contiguous, so reshape is a metadata-only operation.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _relu_flatten_impl(in_0):
    # in_0 is relu's output: shape [B, 2048, 1, 1], all values ≥ 0.
    # dropout(p=0,train=False) is identity; flatten = reshape to [B, 2048].
    batch = in_0.shape[0]
    return in_0.reshape(batch, -1)


# ---------------------------------------------------------------------------
# replacement_func – zero-arg function returning the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return _relu_flatten_impl