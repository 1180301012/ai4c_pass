"""
Fused Edge Normalization pass – gather+multiply kernel.

In the compiled (Dynamo-traced) FX graph, the 4-node gather+multiply subgraph is:
    tmp_5 = tmp_2[in_5]        (getitem, row indices)
    tmp_6 = tmp_5 * in_4       (mul, edge weights)
    tmp_7 = tmp_2[in_2]        (getitem, col indices)
    tmp_8 = tmp_6 * tmp_7      (mul, final norm)

We match exactly this subgraph with tmp_2 as a PLACEHOLDER input.
This avoids:
  - NOT_CONTAINED (pow_ leaks to eq/masked_fill_)
  - OP_MISMATCH  (__eq__ dunder: Dynamo=call_method, FX=call_function)
  - TARGET_MISMATCH (.eq() gives 'eq' but actual is '__eq__')

At runtime, tmp_2 already holds the masked degree normalisation
(pow_(-0.5) + masked_fill_ ran in-place before this subgraph).

The replacement fuses 4 CUDA kernel launches into 1 Triton kernel:
    out[e] = tmp_2[row[e]] * ew[e] * tmp_2[col[e]]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused gather + element-wise multiply
# ---------------------------------------------------------------------------

@triton.jit
def gather_mul_kernel(
    norm_ptr,         # [N_nodes]  masked degree norms  (fp16/bf16)
    row_ptr,          # [N_edges]  source-node indices  (int64)
    col_ptr,          # [N_edges]  dest-node   indices  (int64)
    ew_ptr,           # [N_edges]  edge weights         (fp16/bf16)
    out_ptr,          # [N_edges]  output               (fp16/bf16)
    N_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_edges

    row_idx  = tl.load(row_ptr + offsets, mask=mask, other=0)
    col_idx  = tl.load(col_ptr + offsets, mask=mask, other=0)
    ew       = tl.load(ew_ptr  + offsets, mask=mask, other=0.0)

    norm_row = tl.load(norm_ptr + row_idx, mask=mask, other=0.0)
    norm_col = tl.load(norm_ptr + col_idx, mask=mask, other=0.0)

    out_f32  = norm_row.to(tl.float32) * ew.to(tl.float32) * norm_col.to(tl.float32)
    tl.store(out_ptr + offsets, out_f32.to(ew.dtype), mask=mask)


# ---------------------------------------------------------------------------
# @torch.fx.wrap wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_gather_mul(tmp_2, in_5, in_4, in_2):
    """
    Fused gather + multiply for edge normalisation.

    tmp_2 : [N_nodes] masked degree norms  (fp16/bf16, CUDA)
    in_5  : [N_edges] row (source) indices (int64, CUDA)
    in_4  : [N_edges] edge weights         (fp16/bf16, CUDA)
    in_2  : [N_edges] col (dest)   indices (int64, CUDA)
    """
    N_edges   = in_5.shape[0]
    out       = torch.empty(N_edges, dtype=in_4.dtype, device=in_4.device)
    # BLOCK_SIZE=64 (2 warps/block) balanced between SM utilisation and warp efficiency
    # GAE: N_edges=1100 → 18 blocks | RECT_L: N_edges=256 → 4 blocks
    BLOCK     = 64
    n_blocks  = (N_edges + BLOCK - 1) // BLOCK
    gather_mul_kernel[(n_blocks,)](
        tmp_2, in_5, in_2, in_4, out, N_edges,
        BLOCK_SIZE=BLOCK,
        num_warps=2,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(tmp_2, in_5, in_4, in_2):
    """
    4-node gather+multiply sub-graph.

    tmp_2 is a PLACEHOLDER – avoids NOT_CONTAINED and __eq__ OP/TARGET_MISMATCH.

    tmp_2 = pre-computed masked degree norms  [N_nodes]  fp16/bf16  (CUDA)
    in_5  = row indices                       [N_edges]  int64      (CUDA)
    in_4  = edge weights                      [N_edges]  fp16/bf16  (CUDA)
    in_2  = col indices                       [N_edges]  int64      (CUDA)
    """
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_5, in_4, in_2):
    return (tmp_2, in_5, in_4, in_2)


def replacement_func():
    return fused_gather_mul