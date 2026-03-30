import torch
import triton
import triton.language as tl
from torch import inf


@triton.jit
def _gather_mul_kernel(
    norm_ptr,        # already-normalized degree tensor
    row_ptr,         # row indices [n_edges], int64
    col_ptr,         # col indices [n_edges], int64
    edge_weight_ptr, # edge weights [n_edges], fp16/bf16
    out_ptr,         # output [n_edges], fp16/bf16
    n_edges,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: out[e] = norm[row[e]] * edge_weight[e] * norm[col[e]]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges

    row_idx = tl.load(row_ptr + offsets, mask=mask, other=0)
    col_idx = tl.load(col_ptr + offsets, mask=mask, other=0)
    edge_w = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)

    row_norm = tl.load(norm_ptr + row_idx, mask=mask, other=0.0)
    col_norm = tl.load(norm_ptr + col_idx, mask=mask, other=0.0)

    out_f32 = row_norm.to(tl.float32) * edge_w.to(tl.float32) * col_norm.to(tl.float32)

    if IS_BF16:
        tl.store(out_ptr + offsets, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offsets, out_f32.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_gather_mul(in_2, tmp_2, in_4, in_5):
    """
    Replacement for: (tmp_2[in_5] * in_4) * tmp_2[in_2]

    Strategy:
    - For small n_edges (<=512, e.g. RECT_L with 256 edges): fall back to native
      PyTorch ops — Triton kernel overhead exceeds compute for tiny tensors.
    - For larger n_edges (e.g. GAE with 1100 edges): use a single Triton kernel
      that fuses 2 gathers + 2 multiplies into one GPU pass, reducing kernel
      launches from 4 to 1.

    tmp_2 is the already-normalized degree tensor (output of pow_(-0.5) +
    masked_fill_), so we just gather and multiply.
    """
    n_edges = in_4.numel()

    # For very small tensors Triton overhead > savings; use native PyTorch
    if n_edges <= 512:
        return (tmp_2[in_5] * in_4) * tmp_2[in_2]

    # For larger tensors: fuse 4 ops into 1 Triton kernel
    is_bf16 = in_4.dtype == torch.bfloat16
    out = torch.empty_like(in_4)
    BLOCK_SIZE = 256
    n_blocks = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    _gather_mul_kernel[(n_blocks,)](
        tmp_2, in_5, in_2, in_4, out, n_edges,
        IS_BF16=is_bf16,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def pattern(in_2, tmp_2, in_4, in_5):
    """
    Matches: gather(tmp_2, row) * edge_weight * gather(tmp_2, col)

    tmp_2 is a PLACEHOLDER INPUT (already-normalized degrees after pow_ +
    masked_fill_). Those ops run natively — only the 4-op gather/multiply
    chain is replaced by a single Triton kernel (or native fallback).

    Avoids NOT_CONTAINED because tmp_2 is a placeholder (not internal).
    No dead-code nodes.
    """
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(in_2, tmp_2, in_4, in_5):
    return (in_2, tmp_2, in_4, in_5)


def replacement_func():
    return fused_gather_mul