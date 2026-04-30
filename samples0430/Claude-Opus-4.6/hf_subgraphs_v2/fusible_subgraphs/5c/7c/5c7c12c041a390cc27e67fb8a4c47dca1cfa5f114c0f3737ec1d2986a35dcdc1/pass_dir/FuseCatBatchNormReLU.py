import torch
import triton
import triton.language as tl


def pattern(a, b, running_mean, running_var, weight, bias):
    cat_out = torch.cat([a, b], 1)
    bn_out = torch.nn.functional.batch_norm(cat_out, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    result = torch.nn.functional.relu(bn_out, inplace=False)
    return result


def replacement_args(a, b, running_mean, running_var, weight, bias):
    return (a, b, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_cat_bn_relu_kernel(
    a_ptr, b_ptr, output_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    C_a, C_b, HW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_id = tl.program_id(1)
    spatial_block_id = tl.program_id(0)

    C_total = C_a + C_b
    c = nc_id % C_total
    n = nc_id // C_total

    # Spatial offsets for this block
    offsets = spatial_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Load BN parameters (scalar per program instance)
    mean_val = tl.load(running_mean_ptr + c).to(tl.float32)
    var_val = tl.load(running_var_ptr + c).to(tl.float32)
    w_val = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute scale and shift
    inv_std = tl.rsqrt(var_val + 0.001)
    scale = w_val * inv_std
    shift = b_val - mean_val * scale

    # Determine which source tensor this channel belongs to
    is_from_a = c < C_a
    # Safe channel index for b (avoid negative when c < C_a)
    b_c = tl.maximum(c - C_a, 0)

    # Compute base offsets for loading
    a_base = n * C_a * HW + c * HW
    b_base = n * C_b * HW + b_c * HW

    # Load from appropriate source tensor
    x_a = tl.load(a_ptr + a_base + offsets, mask=mask & is_from_a, other=0.0)
    x_b = tl.load(b_ptr + b_base + offsets, mask=mask & (~is_from_a), other=0.0)
    x = tl.where(is_from_a, x_a, x_b).to(tl.float32)

    # Fused batch_norm + relu
    out = scale * x + shift
    out = tl.maximum(out, 0.0)

    # Store to output
    out_base = n * C_total * HW + c * HW
    tl.store(output_ptr + out_base + offsets, out, mask=mask)


@torch.fx.wrap
def fused_cat_bn_relu(a, b, running_mean, running_var, weight, bias):
    N = a.shape[0]
    C_a = a.shape[1]
    C_b = b.shape[1]
    H = a.shape[2]
    W = a.shape[3]
    C_total = C_a + C_b
    HW = H * W

    output = torch.empty((N, C_total, H, W), dtype=a.dtype, device=a.device)

    grid = lambda meta: (
        (HW + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
        N * C_total,
    )

    fused_cat_bn_relu_kernel[grid](
        a, b, output,
        running_mean, running_var,
        weight, bias,
        C_a, C_b, HW,
    )

    return output


def replacement_func():
    return fused_cat_bn_relu