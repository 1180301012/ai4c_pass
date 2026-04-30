import torch
import triton
import triton.language as tl


def pattern(cat_input, running_mean, running_var, weight, bias):
    bn_result = torch.nn.functional.batch_norm(cat_input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_result = torch.nn.functional.silu(bn_result, inplace=True)
    mean_result = silu_result.mean((2, 3), keepdim=True)
    return (silu_result, mean_result)


def replacement_args(cat_input, running_mean, running_var, weight, bias):
    return (cat_input, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_bn_silu_mean_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    silu_out_ptr, mean_out_ptr,
    N, C, H, W, spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Load BN parameters (compute in float32 for precision)
    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    wt = tl.load(weight_ptr + c).to(tl.float32)
    bi = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute BN: y = scale * x + offset
    # where scale = weight / sqrt(var + eps), offset = bias - mean * scale
    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale = wt * inv_std
    offset = bi - rm * scale

    # Base offset for this (n, c) in contiguous [N, C, H, W] layout
    base = n * C * spatial_size + c * spatial_size

    # Iterate over spatial positions
    accum = 0.0

    for start in range(0, spatial_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size

        # Load input element (cast to float32 for computation)
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

        # Batch normalization: y = scale * x + offset
        bn_val = scale * x + offset

        # For invalid elements, set to 0 for numerical safety in sigmoid/exp
        bn_val_safe = tl.where(mask, bn_val, 0.0)

        # SiLU activation: x * sigmoid(x)
        silu_val = bn_val_safe * tl.sigmoid(bn_val_safe)

        # Store silu output
        tl.store(silu_out_ptr + base + offs, silu_val, mask=mask)

        # Accumulate for mean (invalid elements contribute 0)
        accum += tl.sum(silu_val, axis=0)

    # Compute mean over spatial dimensions
    mean_val = accum / spatial_size

    # Store mean output [N, C, 1, 1] in contiguous layout
    tl.store(mean_out_ptr + n * C + c, mean_val)


@torch.fx.wrap
def fused_bn_silu_mean(cat_input, running_mean, running_var, weight, bias):
    N, C, H, W = cat_input.shape
    eps = 1e-05
    spatial_size = H * W

    silu_out = torch.empty_like(cat_input)
    mean_out = torch.empty((N, C, 1, 1), dtype=cat_input.dtype, device=cat_input.device)

    grid = (N * C,)

    fused_bn_silu_mean_kernel[grid](
        cat_input, running_mean, running_var, weight, bias,
        silu_out, mean_out,
        N, C, H, W, spatial_size,
        eps=eps,
    )

    return (silu_out, mean_out)


def replacement_func():
    return fused_bn_silu_mean