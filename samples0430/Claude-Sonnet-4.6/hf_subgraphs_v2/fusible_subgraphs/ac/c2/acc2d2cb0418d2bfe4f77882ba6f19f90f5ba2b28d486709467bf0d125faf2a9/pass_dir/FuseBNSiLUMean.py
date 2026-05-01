import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu = torch.nn.functional.silu(bn, inplace=True)
    mean = silu.mean((2, 3), keepdim=True)
    return silu, mean


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}),
        triton.Config({'BLOCK_HW': 64}),
        triton.Config({'BLOCK_HW': 128}),
        triton.Config({'BLOCK_HW': 256}),
        triton.Config({'BLOCK_HW': 512}),
        triton.Config({'BLOCK_HW': 1024}),
        triton.Config({'BLOCK_HW': 2048}),
    ],
    key=['HW'],
)
@triton.jit
def bn_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one (n, c) slice of shape [HW].
    Computes: out = silu(batch_norm(x))
    Also accumulates sum for mean over spatial dims.
    """
    nc_idx = tl.program_id(0)
    c_idx = nc_idx % C

    # Load per-channel BN parameters (upcast to fp32 for precision)
    rmean = tl.load(running_mean_ptr + c_idx).to(tl.float32)
    rvar  = tl.load(running_var_ptr  + c_idx).to(tl.float32)
    w     = tl.load(weight_ptr       + c_idx).to(tl.float32)
    b     = tl.load(bias_ptr         + c_idx).to(tl.float32)

    # Fuse BN into: out = x * scale + shift
    inv_std = 1.0 / tl.math.sqrt(rvar + eps)
    scale   = w * inv_std
    shift   = b - rmean * scale

    base = nc_idx * HW
    total_sum = 0.0

    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        # Load in native dtype, upcast for compute
        x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        x_f32 = x_val.to(tl.float32)

        # BatchNorm (inference)
        x_norm = x_f32 * scale + shift

        # SiLU: x * sigmoid(x)
        silu_f32 = x_norm * tl.sigmoid(x_norm)

        # Store in original dtype
        tl.store(out_ptr + base + offsets, silu_f32.to(x_val.dtype), mask=mask)

        # Accumulate for mean (fp32)
        total_sum = total_sum + tl.sum(tl.where(mask, silu_f32, 0.0))

    # Write mean for this (n,c) position; cast back to original dtype
    mean_val = total_sum / HW
    # Re-load a single element to recover original dtype for the cast
    x0 = tl.load(x_ptr + base, mask=True, other=0.0)
    tl.store(mean_out_ptr + nc_idx, mean_val.to(x0.dtype))


@torch.fx.wrap
def fused_bn_silu_mean(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out          = torch.empty_like(x)
    mean_out_flat = torch.empty((NC,), dtype=x.dtype, device=x.device)

    bn_silu_mean_kernel[(NC,)](
        x, running_mean, running_var, weight, bias,
        out, mean_out_flat,
        C, HW,
        eps=1e-05,
    )

    return out, mean_out_flat.view(N, C, 1, 1)


def replacement_func():
    return fused_bn_silu_mean