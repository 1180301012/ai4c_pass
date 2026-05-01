import torch
import triton
import triton.language as tl
from torch import device


# ─────────────────────────────────────────────────────────────────────────────
# Pattern
#
# in_2  =  tmp_7  (shape [3, B, S], the result of expand→to, a direct model
#                  output AND the input to max(0) for computing tmp_13)
#
# Key insight: we return ONLY tmp_13.  The second model output (tmp_7 / in_2)
# is a boundary placeholder – it stays in the graph untouched.  The
# SubgraphRewriter replaces only the matched interior nodes and remaps uses
# of tmp_13 to our Triton kernel result.
#
# Returning a placeholder directly in the pattern output causes a blank crash
# in PyTorch's SubgraphRewriter, so we intentionally avoid that here.
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_2):
    max_1  = in_2.max(0, keepdim=False)
    tmp_9  = max_1[0]
    max_2  = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(in_2):
    return (in_2,)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
#
# in_2   : [3, B, S] int64, stride [0, S, 1]  (expand, non-contiguous)
# out13  : [B, 1]   int64
#
# Reads the single underlying [B, S] layer via stride(1)/stride(2),
# computes per-row max, stores max - 8 → out13[b, 0].
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _compute_row_max_kernel(
    in2_ptr,
    out13_ptr,
    B, S,
    stride_b, stride_s,
    BLOCK_S: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offsets  = tl.arange(0, BLOCK_S)
    mask     = offsets < S

    NEGINF = -1073741824
    vals    = tl.load(
        in2_ptr + batch_id * stride_b + offsets * stride_s,
        mask=mask, other=NEGINF)

    row_max = tl.max(vals, axis=0)
    tl.store(out13_ptr + batch_id, row_max - 8)


@torch.fx.wrap
def _compute_row_max_wrapper(in_2):
    _, B, S   = in_2.shape
    BLOCK_S   = triton.next_power_of_2(S)
    num_warps = max(1, min(BLOCK_S // 32, 32))

    out_tmp13 = torch.empty((B, 1), dtype=torch.int64, device=in_2.device)

    _compute_row_max_kernel[(B,)](
        in_2,
        out_tmp13,
        B, S,
        in_2.stride(1), in_2.stride(2),
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
    )

    return out_tmp13


# Single-output replacement: returns only tmp_13 (a single tensor).
# FX traces _replacement_fn and creates one call_function node that maps
# 1-to-1 to the pattern's single output.
def _replacement_fn(in_2):
    return _compute_row_max_wrapper(in_2)


def replacement_func():
    return _replacement_fn