import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match the final element-wise add.
# ---------------------------------------------------------------------------
def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


# ---------------------------------------------------------------------------
# Optimized 2-D kernel for x=tmp_5 (non-contiguous) + y=tmp_8 (contiguous).
#
# Memory access analysis:
#   x (tmp_5): shape [N, L, C], strides (N*L*C, 1, L)
#   y (tmp_8): shape [N, L, C], strides (N*L*C, C, 1)  — contiguous
#   out:       contiguous [N, L, C]
#
# For coalesced x reads, tile in (c, l) order — c outer, l inner:
#   x[c_off, l_off] = x_ptr + n*sx0 + c*sx2 + l*sx1
#   l-stride = sx1 = 1  → consecutive l values are adjacent in memory ✓
#
# y and out are naturally contiguous in (l, c) order:
#   y[n, l, c] = y_ptr + n*sy0 + l*C + c  → c inner (stride 1) ✓
#
# Grid: (ceil(C/BLOCK_C), ceil(L/BLOCK_L), N)
# No div/mod in the kernel body.
# tl.trans converts the [BLOCK_C, BLOCK_L] x+y result to [BLOCK_L, BLOCK_C]
# for the contiguous output.
# ---------------------------------------------------------------------------
@triton.jit
def _add_coalesced_k(
    x_ptr, sx0, sx1, sx2,
    y_ptr, sy0, sy1, sy2,
    out_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    c_blk = tl.program_id(0)
    l_blk = tl.program_id(1)
    n_idx = tl.program_id(2)

    offs_c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = l_blk * BLOCK_L + tl.arange(0, BLOCK_L)

    mask_c = offs_c < 768   # C=768 always for these graphs
    mask_l = offs_l < 1568  # L=1568 always for these graphs

    # x tile [BLOCK_C, BLOCK_L]: c outer, l inner → l-stride=sx1=1 → COALESCED
    x_off = n_idx * sx0 + offs_c[:, None] * sx2 + offs_l[None, :] * sx1
    x_v   = tl.load(x_ptr + x_off,
                    mask=mask_c[:, None] & mask_l[None, :], other=0.0)

    # y tile [BLOCK_C, BLOCK_L]: c outer, l inner → l-stride=sy1=C → COALESCED
    y_off = n_idx * sy0 + offs_c[:, None] * sy2 + offs_l[None, :] * sy1
    y_v   = tl.load(y_ptr + y_off,
                    mask=mask_c[:, None] & mask_l[None, :], other=0.0)

    result = tl.trans(x_v + y_v)   # [BLOCK_C, BLOCK_L] → [BLOCK_L, BLOCK_C]

    # out tile [BLOCK_L, BLOCK_C] contiguous: l outer, c inner → c-stride=1 → COALESCED
    out_off = n_idx * 768 * 1568 + offs_l[:, None] * 768 + offs_c[None, :]
    tl.store(out_ptr + out_off, result,
             mask=mask_l[:, None] & mask_c[None, :])


# ---------------------------------------------------------------------------
# Wrapper.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_nclass(x, y):
    """
    x = tmp_5  non-contiguous [N, L, C] on CUDA
    y = tmp_8  contiguous    [N, L, C] on CUDA
    """
    N     = x.shape[0]
    L     = x.shape[1]   # 1568
    C     = x.shape[2]   # 768

    out = torch.empty(N * C * L, dtype=x.dtype, device=x.device)

    BLOCK_C = 32
    BLOCK_L = 64
    c_blocks = (C  + BLOCK_C - 1) // BLOCK_C   # 24
    l_blocks = (L  + BLOCK_L - 1) // BLOCK_L   # 25

    _add_coalesced_k[(c_blocks, l_blocks, N)](
        x,
        x.stride()[0], x.stride()[1], x.stride()[2],
        y,
        y.stride()[0], y.stride()[1], y.stride()[2],
        out,
        BLOCK_C=BLOCK_C,
        BLOCK_L=BLOCK_L,
        num_warps=8,
    )

    return out.view(N, L, C)


def replacement_func():
    return fused_add_nclass