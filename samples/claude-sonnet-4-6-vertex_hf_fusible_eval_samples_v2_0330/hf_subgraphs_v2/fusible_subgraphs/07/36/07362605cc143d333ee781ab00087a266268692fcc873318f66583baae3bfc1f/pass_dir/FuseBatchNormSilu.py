import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,  'num_warps': 2}),
        triton.Config({'BLOCK_HW': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_HW': 256, 'num_warps': 4}),
        triton.Config({'BLOCK_HW': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_HW': 64,  'num_warps': 4}),
        triton.Config({'BLOCK_HW': 128, 'num_warps': 2}),
        triton.Config({'BLOCK_HW': 256, 'num_warps': 8}),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_bn_silu_kernel(
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
    """
    Fused inference-mode batch_norm + SiLU kernel.
    Grid: (C, ceil(HW / BLOCK_HW))
    Each program handles a contiguous slice of HW spatial elements for one channel.
    """
    pid_c  = tl.program_id(0)   # channel index
    pid_hw = tl.program_id(1)   # tile index along HW

    # ---- Per-channel statistics (scalar loads) ----
    mean = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var  = tl.load(var_ptr    + pid_c).to(tl.float32)
    w    = tl.load(weight_ptr + pid_c).to(tl.float32)
    b    = tl.load(bias_ptr   + pid_c).to(tl.float32)

    # Precompute affine coefficients: y = x * scale + shift
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = w * inv_std
    shift   = b - mean * scale

    # ---- HW tile ----
    hw_start = pid_hw * BLOCK_HW
    offsets  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = offsets < HW

    base = pid_c * HW
    x    = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    x_f  = x.to(tl.float32)

    # Batch norm (inference)
    x_norm = x_f * scale + shift

    # SiLU: x * sigmoid(x)
    x_out = x_norm * tl.sigmoid(x_norm)

    tl.store(out_ptr + base + offsets, x_out.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_bn_silu(x, running_mean, running_var, weight, bias):
    """
    Drop-in replacement for:
        y = batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-5)
        return silu(y)
    x is expected to have shape [N, C, H, W] (N=1 in these graphs).
    """
    N  = x.shape[0]
    C  = x.shape[1]
    HW = x.numel() // (N * C)

    out = torch.empty_like(x)

    device = x.device
    dtype  = x.dtype

    # BN parameters may live on CPU – move them to the same device/dtype as x
    rm = running_mean.to(device=device, dtype=dtype)
    rv = running_var.to(device=device,  dtype=dtype)
    wt = weight.to(device=device,       dtype=dtype)
    bt = bias.to(device=device,         dtype=dtype)

    grid = lambda meta: (C, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_bn_silu_kernel[grid](
        x, rm, rv, wt, bt, out,
        C, HW,
        1e-5,          # eps
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API consumed by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias):
    """
    Match: inference batch_norm followed by SiLU (inplace).
    Matches both the 256-channel and 512-channel variants because the
    reshape that precedes batch_norm is a free view and is not included here.
    """
    tmp_5 = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_bn_silu