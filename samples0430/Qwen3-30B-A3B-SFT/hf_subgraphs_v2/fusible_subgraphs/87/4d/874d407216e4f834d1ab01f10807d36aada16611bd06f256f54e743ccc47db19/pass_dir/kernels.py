"""
Shared Triton kernel and dispatch wrapper for broadcast-multiply passes.

Pattern (each pass matches this specific subgraph):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2

Two pass files differentiate by the route string:
  "1100_16"  → N=1100, D=16  (bfloat16 / float32, GAE graphs)
  "256_128"  → N=256,  D=128 (float16,        RECT_L graph)

Kernel design:
  - One program per row (pid = row index, 0..N-1).
  - BLOCK_SIZE == D (power-of-2 for both shapes).
  - No integer-division inside the kernel (cleaner, no div instruction).
  - num_warps chosen by heuristic (not autotune) to eliminate per-call
    benchmarking overhead.  1 warp for D=16, 4 warps for D=128.

The replacement returns the single tensor tmp_1 — no tuple return that
confuses FX's subgraph rewriter.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: broadcast multiply  in_1[N] × in_2[N,D] → out[N,D]
#   One program per row; BLOCK_SIZE == D.
# ---------------------------------------------------------------------------
@triton.jit
def _broadcast_mul_row_kernel(
    in1_ptr,               # [N]     stride-1
    in2_ptr,               # [N, D]  contiguous row-major
    out_ptr,               # [N, D]  contiguous row-major
    BLOCK_SIZE: tl.constexpr,   # == D
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    in1_val = tl.load(in1_ptr + pid)
    in2_vals = tl.load(in2_ptr + pid * BLOCK_SIZE + cols)
    tl.store(out_ptr + pid * BLOCK_SIZE + cols, in1_val * in2_vals)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper — IDENTICAL across both pass files.
# Returns the single tensor tmp_1 (no tuple) to stay FX-compatible.
#
# Using in_2.shape dynamically ensures the correct output shape for every
# graph variant:
#   GAE (bfloat16 / float32): in_2=[1100,16]  → out=[1100,16]
#   RECT_L (float16):         in_2=[256,128]  → out=[256,128]
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_fused_view_mul(in_1, in_2, route):
    N = in_2.shape[0]
    D = in_2.shape[1]
    out = torch.empty((N, D), dtype=in_2.dtype, device=in_2.device)
    # One program per row; BLOCK_SIZE == D (constexpr, power-of-2).
    # num_warps: 1 warp for D=16 (GAE), 4 warps for D=128 (RECT_L).
    num_warps = 1 if D <= 16 else 4
    _broadcast_mul_row_kernel[(N,)](
        in_1, in_2, out, BLOCK_SIZE=D, num_warps=num_warps
    )
    return out