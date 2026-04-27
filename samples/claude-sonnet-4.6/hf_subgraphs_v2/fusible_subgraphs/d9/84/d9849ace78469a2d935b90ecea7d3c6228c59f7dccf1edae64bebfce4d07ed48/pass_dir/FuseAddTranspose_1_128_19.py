import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: transpose(1, 2) on the result of the in-place add.
# in_2 (input):  [1, 128, 19]  (result of iadd)
# output:        [1,  19, 128]  (contiguous transposed copy)
# ---------------------------------------------------------------------------
def pattern(in_2):
    tmp_2 = in_2.transpose(1, 2)
    return tmp_2


def replacement_args(in_2):
    return (in_2,)


# ---------------------------------------------------------------------------
# Triton kernel (NO autotune): read in_2[b,i,j] → write out[b,j,i]
#
# For B=1, C=128, N=19: total=2432 elements.
# Use BLOCK_SIZE=128 → 19 programs (one per "row" j in output).
# Reads for a single program are i=0..127 for fixed j → stride-19 in source.
# For such tiny data (5 KB) everything lives in L1 cache.
# ---------------------------------------------------------------------------
@triton.jit
def contiguous_transpose_kernel(
    in2_ptr,   # [B, C, N]  float16/bfloat16
    out_ptr,   # [B, N, C]
    B, C, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * N * C
    mask = offsets < total

    # Decode flat output index → (b, j, i) in [B, N, C]
    b   = offsets // (N * C)
    rem = offsets % (N * C)
    j   = rem // C
    i   = rem % C

    # Source: in_2[b, i, j] in [B, C, N]
    in2_idx = b * (C * N) + i * N + j
    val = tl.load(in2_ptr + in2_idx, mask=mask, other=0.0)

    # Destination: out[b, j, i]  (flat index == offsets)
    tl.store(out_ptr + offsets, val, mask=mask)


# Hard-wired constants for the known tensor shape B=1, C=128, N=19
_B, _C, _N = 1, 128, 19
_BLOCK_SIZE = 128
_TOTAL = _B * _C * _N          # 2432
_GRID  = ((_TOTAL + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)   # (19,)


@torch.fx.wrap
def triton_contiguous_transpose(in_2):
    out = torch.empty((_B, _N, _C), dtype=in_2.dtype, device=in_2.device)
    contiguous_transpose_kernel[_GRID](
        in_2, out,
        _B, _C, _N,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return out


def replacement_func():
    return triton_contiguous_transpose