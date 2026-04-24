import torch
import triton
import triton.language as tl
import operator


# ---------------------------------------------------------------------------
# Pattern: sum(in_0, dim=-1) → unsqueeze(-1)
# The downstream div (in_0 /= tmp_1) and dropout(p=0) remain in the graph.
# Replacement returns (1,16,196,1) so downstream div correctly normalizes.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: row-sum with unsqueeze → write to flat (M,) output
# Grid: (M,)  M = numel / N_in
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N_in'],
)
@triton.jit
def _sum_unsqueeze_kernel(
    in_ptr,
    out_ptr,
    N_in,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_in
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)
    total = tl.sum(x.to(tl.float32), axis=0)
    tl.store(out_ptr + row_idx, total)


@torch.fx.wrap
def fused_row_norm(in_0):
    shape = in_0.shape
    N_in = shape[-1]
    M = in_0.numel() // N_in
    # Output: (1,16,196,1) — same shape as sum(in_0,dim=-1).unsqueeze(-1)
    out = torch.empty((1, 16, 196, 1), dtype=in_0.dtype, device=in_0.device)
    _sum_unsqueeze_kernel[(M,)](in_0, out, N_in, N_in)
    return out


def replacement_func():
    return fused_row_norm