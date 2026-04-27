import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: layer-scale multiply + residual add
# dropout(p=0, training=False) is a strict no-op → excluded
# batch_norm runs unchanged after this pass
# ---------------------------------------------------------------------------
def pattern(conv_out, gamma, residual):
    scaled = conv_out * gamma
    pre_bn = residual + scaled
    return pre_bn


def replacement_args(conv_out, gamma, residual):
    return (conv_out, gamma, residual)


# ---------------------------------------------------------------------------
# Triton kernel  –  2-D grid (B*C, HW-tiles)
#   • channel index derived ONCE per block (not per element)
#   • gamma loaded as a SCALAR per block (no gather, perfect caching)
#   • spatial loads perfectly coalesced
# ---------------------------------------------------------------------------
@triton.jit
def _scale_add_k1024(
    x_ptr, gamma_ptr, res_ptr, out_ptr,
    HW, C,
    BLOCK_HW: tl.constexpr,
):
    bc_pid  = tl.program_id(0)
    hw_tile = tl.program_id(1)
    c       = bc_pid % C
    base    = bc_pid * HW
    hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW
    offs    = base + hw_offs
    gval = tl.load(gamma_ptr + c)
    x    = tl.load(x_ptr   + offs, mask=mask, other=0.)
    r    = tl.load(res_ptr  + offs, mask=mask, other=0.)
    tl.store(out_ptr + offs, r + x * gval, mask=mask)


@triton.jit
def _scale_add_k2048(
    x_ptr, gamma_ptr, res_ptr, out_ptr,
    HW, C,
    BLOCK_HW: tl.constexpr,
):
    bc_pid  = tl.program_id(0)
    hw_tile = tl.program_id(1)
    c       = bc_pid % C
    base    = bc_pid * HW
    hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW
    offs    = base + hw_offs
    gval = tl.load(gamma_ptr + c)
    x    = tl.load(x_ptr   + offs, mask=mask, other=0.)
    r    = tl.load(res_ptr  + offs, mask=mask, other=0.)
    tl.store(out_ptr + offs, r + x * gval, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper – opaque FX leaf
# HW > 4096 → BLOCK_HW=2048, num_warps=8  (large spatial, better L2 reuse)
# HW ≤ 4096 → BLOCK_HW=1024, num_warps=4  (high occupancy on A30)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_scale_add(conv_out, gamma, residual):
    B, C, H, W = conv_out.shape
    HW  = H * W
    BC  = B * C
    out = torch.empty_like(conv_out)

    if HW > 4096:
        BLOCK_HW = 2048
        hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW
        _scale_add_k2048[(BC, hw_tiles)](
            conv_out, gamma, residual, out, HW, C,
            BLOCK_HW=BLOCK_HW, num_warps=8,
        )
    else:
        BLOCK_HW = 1024
        hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW
        _scale_add_k1024[(BC, hw_tiles)](
            conv_out, gamma, residual, out, HW, C,
            BLOCK_HW=BLOCK_HW, num_warps=4,
        )

    return out


def replacement_func():
    return triton_scale_add