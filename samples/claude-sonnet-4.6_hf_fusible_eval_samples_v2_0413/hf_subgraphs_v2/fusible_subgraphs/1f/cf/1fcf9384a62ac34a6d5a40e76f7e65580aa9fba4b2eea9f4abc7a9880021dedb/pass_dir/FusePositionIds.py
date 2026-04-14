import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: exact replica of model.py forward (minus the `= None` cleanups)
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
# Fused Triton kernel – 1-D, one CTA per row, int64 throughout.
#
#   m   = (x != 1).int64      -- 1 = non-padding, 0 = padding
#   cs  = inclusive_cumsum(m) -- running non-padding count along the row
#   out = cs * m + 1          -- non-pad: 1-indexed position; pad: 1
#
# num_warps = BLOCK_SIZE // 32 gives one hardware warp per 32-element chunk,
# which is the natural mapping for the tree-based warp-level scan inside
# tl.cumsum and minimises unnecessary barrier synchronisation.
# ---------------------------------------------------------------------------
@triton.jit
def _position_ids_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets   = tl.arange(0, BLOCK_SIZE)
    valid     = offsets < seq_len

    row_start = batch_idx * seq_len
    # Load int64; OOB slots filled with 1 → ne(1)==False → m=0 (neutral element)
    x = tl.load(input_ptr + row_start + offsets, mask=valid, other=1)

    # Stay in int64 throughout – avoids any int32 intermediate / final cast
    m      = (x != 1).to(tl.int64)
    cs     = tl.cumsum(m, axis=0)   # inclusive prefix-sum
    result = cs * m + 1             # int64 result

    tl.store(output_ptr + row_start + offsets, result, mask=valid)


# ---------------------------------------------------------------------------
# Wrapper – @torch.fx.wrap so the graph rewriter can call it.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def position_ids_triton(in_0):
    B, S  = in_0.shape
    output = torch.empty_like(in_0)   # int64, same shape as in_0

    # Next power-of-2 >= S, minimum 32 (one full warp)
    BLOCK_SIZE = max(32, 1 << (S - 1).bit_length())
    # Natural warp mapping: one warp per 32 elements
    num_warps  = max(1, BLOCK_SIZE >> 5)

    _position_ids_kernel[(B,)](
        in_0, output, S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output


def replacement_func():
    return position_ids_triton