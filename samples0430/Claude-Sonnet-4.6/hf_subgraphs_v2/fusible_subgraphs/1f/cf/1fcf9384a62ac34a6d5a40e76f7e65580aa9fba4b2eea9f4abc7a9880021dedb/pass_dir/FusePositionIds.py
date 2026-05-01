import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the exact op sequence in every model.py
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
# Triton kernel: one program (block) per row.
#   • loads int64 input tokens
#   • computes int32 mask  (token != 1)
#   • inclusive prefix-sum via tl.cumsum (1-D along the row)
#   • zeroes out padding positions  (prefix * mask)
#   • adds 1 and stores as int64
# ---------------------------------------------------------------------------
@triton.jit
def _position_ids_kernel(
    in_ptr,
    out_ptr,
    N,           # actual sequence length (runtime)
    stride_row,  # row stride of the input tensor
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid = cols < N

    # Load int64 tokens; out-of-bounds lanes get 1 (= padding token)
    x = tl.load(in_ptr + row * stride_row + cols, mask=valid, other=1)

    # int32 mask: 1 where token != 1, else 0
    m = (x != 1).to(tl.int32)

    # Inclusive prefix sum along the row
    prefix = tl.cumsum(m, axis=0)

    # Zero padding positions, add 1, write as int64
    result = (prefix * m).to(tl.int64) + 1

    tl.store(out_ptr + row * stride_row + cols, result, mask=valid)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _position_ids_fused(in_0):
    B, N = in_0.shape
    out = torch.empty_like(in_0)          # int64, same shape

    # BLOCK_N: next power-of-2 >= N  (must be >= N for correctness)
    BLOCK_N = max(32, 1 << (max(N, 1) - 1).bit_length())

    _position_ids_kernel[(B,)](
        in_0,
        out,
        N,
        in_0.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg function returning a callable
# ---------------------------------------------------------------------------
def replacement_func():
    return _position_ids_fused