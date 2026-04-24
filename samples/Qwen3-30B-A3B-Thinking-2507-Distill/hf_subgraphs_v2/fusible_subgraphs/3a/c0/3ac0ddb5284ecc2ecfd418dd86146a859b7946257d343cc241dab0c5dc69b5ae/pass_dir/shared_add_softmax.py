"""
Shared Triton kernel and dispatch function for fused add+softmax passes.
Both pass files import `fused_add_softmax_dispatch` from here so that
replacement_func() returns the SAME function object, satisfying the
output_pass_replacement_func_limit.

Pattern matched:  add + softmax(dim=-1)   (views and dropout stay in graph)
  - in_0 : [1, 1, N, W]
  - in_1 : [1, 8, N, W]
  - out  : [1, 8, N, W]  (= softmax(in_1 + in_0))
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['W'],
)
@triton.jit
def _fused_add_softmax_kernel(
    in0_ptr,   # [1, 1, N, W]  – batch dim 1 broadcasts over in1's batch dim 8
    in1_ptr,   # [1, 8, N, W]
    out_ptr,   # [1, 8, N, W]  written as flat [8*N*W] under the hood
    N, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One CTA per (batch_b, row_r) pair.
    Computes: out[0, b, r, :] = softmax(in1[0, b, r, :] + in0[0, 0, r, :])
    """
    row = tl.program_id(0)
    b = row // N
    r = row % N

    in1_base = b * N * W + r * W
    in0_base = r * W
    out_base = b * N * W + r * W

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < W

    in1_vals = tl.load(in1_ptr + in1_base + offsets, mask=mask, other=-float('inf'))
    in0_vals = tl.load(in0_ptr + in0_base + offsets, mask=mask, other=0.0)

    x = in1_vals.to(tl.float32) + in0_vals.to(tl.float32)

    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_out = x_exp / x_sum

    tl.store(out_ptr + out_base + offsets, softmax_out, mask=mask)


@torch.fx.wrap
def fused_add_softmax_dispatch(in_0, in_1):
    """
    Fused add+softmax: returns softmax(in_1 + in_0) with same shape as in_1.
    Shape is inferred at runtime from in_1.
    Views and dropout(p=0.0) remain in the surrounding graph as no-ops.
    """
    B  = in_1.shape[1]   # 8
    H  = in_1.shape[2]   # N (300 or 625)
    W  = in_1.shape[3]   # W (625)
    N  = in_1.shape[2]   # same as H for our case

    out = torch.empty((1, B, H, W), dtype=in_1.dtype, device=in_1.device)

    _fused_add_softmax_kernel[(B * N,)](
        in_0, in_1, out,
        N, W,
    )

    return out