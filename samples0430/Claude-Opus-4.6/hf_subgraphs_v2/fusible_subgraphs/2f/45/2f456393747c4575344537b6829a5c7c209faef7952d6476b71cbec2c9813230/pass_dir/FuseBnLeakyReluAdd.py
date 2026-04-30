import torch
import triton
import triton.language as tl


def pattern(conv_out, running_mean, running_var, weight, bias, residual):
    bn = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu = torch.nn.functional.leaky_relu(bn, 0.01, True)
    result = relu + residual
    return result


def replacement_args(conv_out, running_mean, running_var, weight, bias, residual):
    return (conv_out, running_mean, running_var, weight, bias, residual)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_bn_leaky_relu_add_kernel(
    conv_out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    spatial_size,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C

    # Load channel parameters (scalar per channel)
    mean_val = tl.load(mean_ptr + c).to(tl.float32)
    var_val = tl.load(var_ptr + c).to(tl.float32)
    gamma = tl.load(weight_ptr + c).to(tl.float32)
    beta = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute batch norm scale and shift
    # BN: (x - mean) / sqrt(var + eps) * gamma + beta = x * scale + shift
    inv_std = 1.0 / tl.sqrt(var_val + 1e-5)
    scale = gamma * inv_std
    shift = beta - mean_val * scale

    # Base offset for this (n, c) pair in the flattened NCHW tensor
    base = pid * spatial_size

    # Process spatial elements in blocks
    for start in range(0, spatial_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size

        # Load conv output and residual
        x = tl.load(conv_out_ptr + base + offs, mask=mask).to(tl.float32)
        res = tl.load(residual_ptr + base + offs, mask=mask).to(tl.float32)

        # Fused: BatchNorm + LeakyReLU(0.01) + Add
        x = x * scale + shift
        x = tl.where(x >= 0, x, x * 0.01)
        x = x + res

        # Store result
        tl.store(out_ptr + base + offs, x, mask=mask)


@torch.fx.wrap
def fused_bn_leaky_relu_add(conv_out, running_mean, running_var, weight, bias, residual):
    N, C, H, W = conv_out.shape
    spatial_size = H * W
    NC = N * C

    out = torch.empty_like(conv_out)

    grid = (NC,)

    fused_bn_leaky_relu_add_kernel[grid](
        conv_out, running_mean, running_var, weight, bias, residual, out,
        spatial_size, C,
    )

    return out


def replacement_func():
    return fused_bn_leaky_relu_add