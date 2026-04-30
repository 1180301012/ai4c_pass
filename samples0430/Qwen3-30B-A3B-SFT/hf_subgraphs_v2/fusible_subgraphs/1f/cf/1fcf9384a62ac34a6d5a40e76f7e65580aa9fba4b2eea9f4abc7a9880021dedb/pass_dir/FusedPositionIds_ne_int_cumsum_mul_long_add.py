import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full position-id generation sequence in model.py
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: fused ne(1) → int → cumsum → *mask → long → +1
# One program per row; BLOCK_S covers the entire sequence dimension
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32},   num_warps=1),
        triton.Config({'BLOCK_S': 64},   num_warps=2),
        triton.Config({'BLOCK_S': 128},  num_warps=4),
        triton.Config({'BLOCK_S': 256},  num_warps=4),
        triton.Config({'BLOCK_S': 512},  num_warps=8),
        triton.Config({'BLOCK_S': 1024}, num_warps=16),
    ],
    key=['S'],
)
@triton.jit
def _fused_position_ids_kernel(
    in_ptr,   # int64 [B, S]
    out_ptr,  # int64 [B, S]
    B, S,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_S)
    mask = col_offs < S

    x = tl.load(in_ptr + row * S + col_offs, mask=mask, other=1)
    mask_val = (x != 1).to(tl.int32)
    cs = tl.cumsum(mask_val, axis=0)
    out = (cs * mask_val).to(tl.int64) + 1
    tl.store(out_ptr + row * S + col_offs, out, mask=mask)


@torch.fx.wrap
def fused_position_ids(in_0):
    B, S = in_0.shape
    out = torch.empty_like(in_0)
    _fused_position_ids_kernel[(B,)](in_0, out, B, S)
    return out


# ---------------------------------------------------------------------------
# replacement_func: must return a callable, not call it
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_position_ids