import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_out = torch.nn.functional.silu(bn_out, inplace=True)
    mean_out = silu_out.mean((2, 3), keepdim=True)
    return silu_out, mean_out


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def fused_bn_silu_mean_kernel(
    x_ptr, out_ptr, mean_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C

    # Load BN parameters for this channel
    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute scale and shift: output = input * scale + shift
    invstd = 1.0 / tl.sqrt(rv + 1e-5)
    scale = w * invstd
    shift = b - rm * scale

    # Base pointer for this (n, c) slice - contiguous in memory
    base = pid * HW

    # Process spatial elements and accumulate for mean
    total_sum = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused BN + SiLU
        bn_val = x_val * scale + shift
        silu_val = bn_val * tl.sigmoid(bn_val)

        # Accumulate for mean
        total_sum += tl.sum(tl.where(mask, silu_val, 0.0), axis=0)

        # Store output
        tl.store(out_ptr + base + offsets, silu_val, mask=mask)

    # Compute and store mean: shape [N, C, 1, 1]
    mean_val = total_sum / HW
    tl.store(mean_ptr + pid, mean_val)


@torch.fx.wrap
def fused_bn_silu_mean(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)
    mean_out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)

    grid = (N * C,)

    fused_bn_silu_mean_kernel[grid](
        x, out, mean_out,
        running_mean, running_var, weight, bias,
        C, HW,
    )

    return out, mean_out


def replacement_func():
    return fused_bn_silu_mean