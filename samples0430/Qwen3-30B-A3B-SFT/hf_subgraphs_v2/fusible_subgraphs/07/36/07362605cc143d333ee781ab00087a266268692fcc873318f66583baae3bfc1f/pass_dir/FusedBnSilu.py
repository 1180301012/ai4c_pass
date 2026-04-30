import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: batch_norm (inference) followed by silu activation
# This appears in all 4 target graphs after the reshape.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, x):
    """
    in_0  : running_mean  [C]
    in_1  : running_var   [C]
    in_2  : bias          [C]
    in_3  : weight        [C]
    x     : input tensor  [N, C, H, W]  (already reshaped by the graph)
    """
    tmp_5 = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, x):
    # Order: running_mean, running_var, bias, weight, input
    return (in_0, in_1, in_2, in_3, x)


# ---------------------------------------------------------------------------
# Fused Triton kernel: BN-inference + SiLU, one program per (n, c) slice
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    # One program handles one (n, c) slice → HW contiguous elements
    pid = tl.program_id(0)
    c   = pid % C
    n   = pid // C
    base = (n * C + c) * HW

    # Load scalar BN parameters for this channel (upcast to fp32)
    mean = tl.load(mean_ptr   + c).to(tl.float32)
    var  = tl.load(var_ptr    + c).to(tl.float32)
    w    = tl.load(weight_ptr + c).to(tl.float32)
    b    = tl.load(bias_ptr   + c).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Spatial offsets within this (n, c) slice
    hw_offs = tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW

    # Load, normalise, scale/shift, apply SiLU
    x   = tl.load(x_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
    y   = (x - mean) * inv_std * w + b
    out = y * tl.sigmoid(y)          # SiLU = y * σ(y)

    tl.store(out_ptr + base + hw_offs, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_silu(in_0, in_1, in_2, in_3, x):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    x    : [N, C, H, W]
    """
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)

    _fused_bn_silu_kernel[(N * C,)](
        x, in_0, in_1, in_3, in_2, out,   # weight=in_3, bias=in_2  (BN arg order)
        C, HW, 1e-5,
    )
    return out


def replacement_func():
    return fused_bn_silu