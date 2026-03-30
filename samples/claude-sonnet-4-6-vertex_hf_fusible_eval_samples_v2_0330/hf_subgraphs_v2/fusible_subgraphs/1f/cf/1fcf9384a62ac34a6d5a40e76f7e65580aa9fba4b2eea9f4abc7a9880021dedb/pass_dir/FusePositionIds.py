import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel: computes position IDs in one pass
#
# Equivalent Python logic (per row):
#   mask  = (x != 1).astype(int32)          # ne(1) + int()
#   cs    = cumsum(mask)                      # cumsum(dim=1)
#   out   = (cs * mask).astype(int64) + 1    # type_as, +0, *mask, long, +1
#
# BLOCK_SIZE is always chosen >= seq_len (next power-of-2), so the entire
# row fits in one block and no multi-pass prefix-sum is needed.
# ---------------------------------------------------------------------------
@triton.jit
def fused_position_ids_kernel(
    in_ptr,
    out_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per batch row
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    row_start = row_idx * seq_len

    # Load token IDs; out-of-bounds slots get value 1 (= padding ID) so
    # their contribution to the prefix sum is zero.
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=1)

    # ne(1) -> int32  (0 for padding, 1 for real token)
    ne_mask = (x != 1).to(tl.int32)

    # Inclusive prefix sum along the sequence axis
    cumsum = tl.cumsum(ne_mask, axis=0)

    # Zero out padding slots, cast to int64, then add 1
    result = (cumsum * ne_mask).to(tl.int64) + 1

    tl.store(out_ptr + row_start + offsets, result, mask=mask)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


@torch.fx.wrap
def fused_position_ids(in_0):
    batch_size = in_0.shape[0]
    seq_len    = in_0.shape[1]

    out = torch.empty_like(in_0)  # dtype = torch.int64

    # BLOCK_SIZE must be a compile-time constant (constexpr) and must cover
    # the whole row so we get a correct single-pass prefix sum.
    BLOCK_SIZE = max(16, _next_power_of_2(seq_len))

    # Per-size num_warps tuning:
    #  - tiny/small blocks (≤64): 1 warp, single-warp prefix sum is fastest
    #  - medium blocks (128–256): 2 warps, warp-pair scan wins over sequential
    #  - large blocks (≥512): back to 1 warp – cross-warp sync cost exceeds
    #    the parallelism benefit for these small total workloads
    if BLOCK_SIZE <= 64:
        num_warps = 1
    elif BLOCK_SIZE <= 256:
        num_warps = 2
    else:
        num_warps = 1

    grid = (batch_size,)
    fused_position_ids_kernel[grid](
        in_0,
        out,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
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


def replacement_func():
    return fused_position_ids