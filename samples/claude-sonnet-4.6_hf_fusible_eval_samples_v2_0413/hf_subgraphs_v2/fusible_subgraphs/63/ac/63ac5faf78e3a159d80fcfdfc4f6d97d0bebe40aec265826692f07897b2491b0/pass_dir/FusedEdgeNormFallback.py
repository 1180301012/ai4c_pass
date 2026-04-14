"""
Fallback Fused Edge Normalization pass – matches only the 4-node getitem+mul subgraph.

This runs AFTER FusedEdgeNorm. If FusedEdgeNorm matched the full subgraph this
pass finds nothing. If FusedEdgeNorm failed to match (e.g. eq call_method mismatch),
this pass replaces just the gather+multiply portion.

Pattern (tmp_2 is a placeholder – the pre-computed masked degree tensor):
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
"""

import torch
import triton
import triton.language as tl

# Import the shared kernel from FusedEdgeNorm
from pass_dir.FusedEdgeNorm import edge_gather_mul_kernel


@torch.fx.wrap
def fallback_gather_mul(tmp_2, in_5, in_4, in_2):
    """
    Fallback fused gather + multiply.

    tmp_2 : [N_nodes] pre-computed degree norms (fp16/bf16, CUDA)
    in_5  : [N_edges] row indices  (int64, CUDA)
    in_4  : [N_edges] edge weights (fp16/bf16, CUDA)
    in_2  : [N_edges] col indices  (int64, CUDA)
    """
    N_edges = in_5.shape[0]
    out = torch.empty(N_edges, dtype=in_4.dtype, device=in_4.device)

    grid = lambda META: (triton.cdiv(N_edges, META['BLOCK_SIZE']),)
    edge_gather_mul_kernel[grid](tmp_2, in_5, in_2, in_4, out, N_edges)

    return out


def pattern(tmp_2, in_5, in_4, in_2):
    """
    4-node gather+multiply pattern.
    tmp_2 is a PLACEHOLDER – avoids NOT_CONTAINED and OP_MISMATCH issues.
    """
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_5, in_4, in_2):
    return (tmp_2, in_5, in_4, in_2)


def replacement_func():
    return fallback_gather_mul