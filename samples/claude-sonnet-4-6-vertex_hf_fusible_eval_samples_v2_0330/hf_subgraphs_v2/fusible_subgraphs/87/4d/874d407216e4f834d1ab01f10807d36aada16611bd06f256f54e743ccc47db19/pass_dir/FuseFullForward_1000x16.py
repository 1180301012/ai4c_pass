"""
Fusion pass for the GAE graph variant where the zeros tensor has shape (1000, 16).

Matches:
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_4 = tmp_1.new_zeros((1000, 16))
    --- returns (tmp_1, tmp_4) ---

Note: expand_as is intentionally left OUTSIDE the matched subgraph so that
the pattern only returns a 2-tuple (avoids full-graph assertion issues).

Replaces 2 GPU kernel launches (multiply + zeros) with ONE combined Triton
kernel, saving kernel-dispatch latency for these tiny tensors.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  –  view(-1,1) + multiply + new_zeros((1000,16))
# ---------------------------------------------------------------------------
def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return tmp_1, tmp_4


def replacement_args(in_1, in_2):
    return (in_1, in_2)


# ---------------------------------------------------------------------------
# Combined Triton kernel: broadcast-multiply + zero-fill in one launch.
#
# 1-D grid layout:
#   pid  in  [0, mul_blocks)          → multiply section
#   pid  in  [mul_blocks, total)      → zero-fill section
#
# C is tl.constexpr → offs // C becomes a fast bit-shift for pow-2 widths.
# ---------------------------------------------------------------------------
@triton.jit
def fused_mul_zeros_kernel_1000x16(
    in1_ptr,
    in2_ptr,
    mul_ptr,
    zero_ptr,
    MUL_TOTAL,
    Z_TOTAL,
    C:         tl.constexpr,
    BLOCK_MUL: tl.constexpr,
    BLOCK_Z:   tl.constexpr,
):
    pid        = tl.program_id(0)
    mul_blocks = tl.cdiv(MUL_TOTAL, BLOCK_MUL)

    if pid < mul_blocks:
        offs = pid * BLOCK_MUL + tl.arange(0, BLOCK_MUL)
        mask = offs < MUL_TOTAL
        row  = offs // C
        in1  = tl.load(in1_ptr + row,  mask=mask, other=0.0)
        in2  = tl.load(in2_ptr + offs, mask=mask, other=0.0)
        tl.store(mul_ptr + offs, in1 * in2, mask=mask)
    else:
        z_pid  = pid - mul_blocks
        z_offs = z_pid * BLOCK_Z + tl.arange(0, BLOCK_Z)
        z_mask = z_offs < Z_TOTAL
        tl.store(zero_ptr + z_offs, 0.0, mask=z_mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
_Z_ROWS, _Z_COLS = 1000, 16
_Z_TOTAL         = _Z_ROWS * _Z_COLS   # 16 000
_BLOCK_MUL       = 512
_BLOCK_Z         = 512


@torch.fx.wrap
def fused_mul_zeros_1000x16(in_1, in_2):
    N, C      = in_2.shape
    MUL_TOTAL = N * C

    tmp_1 = torch.empty_like(in_2)
    tmp_4 = torch.empty(_Z_ROWS, _Z_COLS, dtype=in_2.dtype, device=in_2.device)

    mul_blocks   = (MUL_TOTAL + _BLOCK_MUL - 1) // _BLOCK_MUL
    z_blocks     = (_Z_TOTAL  + _BLOCK_Z   - 1) // _BLOCK_Z
    total_blocks = mul_blocks + z_blocks

    fused_mul_zeros_kernel_1000x16[(total_blocks,)](
        in_1, in_2, tmp_1, tmp_4,
        MUL_TOTAL, _Z_TOTAL,
        C,
        BLOCK_MUL=_BLOCK_MUL,
        BLOCK_Z=_BLOCK_Z,
    )

    return tmp_1, tmp_4


def replacement_func():
    return fused_mul_zeros_1000x16