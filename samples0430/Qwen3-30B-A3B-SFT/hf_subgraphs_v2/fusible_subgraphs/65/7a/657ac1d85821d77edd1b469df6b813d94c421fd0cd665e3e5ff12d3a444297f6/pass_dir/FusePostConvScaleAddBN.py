import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: layer-scale-mul + residual-add
# (dropout with p=0/training=False is a no-op and may be absent from graph)
# batch_norm is intentionally OUTSIDE this pattern because the framework's
# _replace_pattern asserts len(returning_nodes)==len(replacement_returning)
# and our kernel replaces a single output (pre_bn) that feeds both batch_norm
# and the model output.
# ---------------------------------------------------------------------------

def pattern(conv_out, gamma, residual):
    scaled = conv_out * gamma
    pre_bn = residual + scaled
    return pre_bn


def replacement_args(conv_out, gamma, residual):
    return (conv_out, gamma, residual)


# ---------------------------------------------------------------------------
# Triton kernel: fused scale-multiply + residual-add
#
#   pre_bn = residual + conv_out * gamma
#
# Grid: (N*C, ceil(HW / BLOCK_HW))
#   dim-0: (batch, channel) pairs
#   dim-1: spatial tiles
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
        triton.Config({'BLOCK_HW': 8192}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _fused_scale_add_kernel(
    conv_out_ptr,   # [N, C, HW]
    gamma_ptr,      # [C, 1, 1]  (layer-scale gamma)
    residual_ptr,   # [N, C, HW]
    pre_bn_ptr,     # [N, C, HW]  output
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    hw_pid = tl.program_id(1)

    c_idx = nc_idx % C

    # Load per-channel gamma scalar (promote to fp32 for accuracy)
    gamma_val = tl.load(gamma_ptr + c_idx).to(tl.float32)

    # Spatial tile
    hw_start   = hw_pid * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask       = hw_offsets < HW

    base     = nc_idx * HW
    elem_off = base + hw_offsets

    # Load in original dtype; promote to fp32 for computation
    conv_val = tl.load(conv_out_ptr + elem_off, mask=mask, other=0.0).to(tl.float32)
    res_val  = tl.load(residual_ptr  + elem_off, mask=mask, other=0.0).to(tl.float32)

    pre_bn = res_val + conv_val * gamma_val

    # Store back in original dtype
    tl.store(pre_bn_ptr + elem_off, pre_bn, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_scale_add(conv_out, gamma, residual):
    N, C, H, W = conv_out.shape
    HW = H * W

    pre_bn = torch.empty_like(conv_out)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_scale_add_kernel[grid](
        conv_out, gamma, residual, pre_bn,
        C, HW,
    )

    return pre_bn


def replacement_func():
    return fused_scale_add