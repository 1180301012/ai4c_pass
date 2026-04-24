import torch
import triton
import triton.language as tl


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
# Fused kernel: one pass over a row.
# S is a runtime int — avoids over-specialisation; Triton JIT caches per BLOCK_SIZE.
# ---------------------------------------------------------------------------
@triton.jit
def fused_position_ids_kernel(
    input_ptr,
    output_ptr,
    S,                          # runtime int — avoids multiple kernel specialisations
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * S
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < S
    # Load row; padding tokens (value == 1) give 0, non-padding give 1
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=1)
    is_not_pad = (x != 1).to(tl.int32)
    # Prefix-sum: cumsum[i] = count of non-pad tokens in [0, i] (inclusive).
    # Padding positions: cumsum stays flat through padding blocks → output 0.
    cumsum = tl.cumsum(is_not_pad, axis=0)
    # Position ID: cumsum[i]+1 for non-pad; 0 for padding (mask handles it).
    output = tl.where(mask, cumsum + 1, 0)
    tl.store(output_ptr + row_start + cols, output.to(tl.int64), mask=mask)


@torch.fx.wrap
def fused_position_ids(in_0):
    B, S = in_0.shape
    output = torch.empty_like(in_0)
    # Compute BLOCK_SIZE as next power-of-2 ≥ S using arithmetic (no branches)
    BLOCK_SIZE = 1 << (max(S, 1).bit_length())
    # num_warps: scale with BLOCK_SIZE for better GPU occupancy
    num_warps = min(8, max(1, BLOCK_SIZE // 64))
    fused_position_ids_kernel[(B,)](in_0, output, S, BLOCK_SIZE=BLOCK_SIZE,
                                    num_warps=num_warps)
    return output


def replacement_func():
    return fused_position_ids