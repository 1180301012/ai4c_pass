import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Pattern to match: BatchNorm + LeakyReLU fusion"""
    bn_output = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    output = torch.nn.functional.leaky_relu(bn_output, 0.01, True)
    return output

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_bn_leaky_relu_kernel_channelwise(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    N, C, spatial_size,
    eps: tl.constexpr,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm + LeakyReLU kernel - channel-wise processing"""
    # Each program processes one (batch, channel) pair
    pid_nc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    # Decode batch and channel
    n = pid_nc // C
    c = pid_nc % C
    
    # Load batch norm parameters once per (N, C) pair
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)
    
    # Precompute normalization factor
    inv_std = tl.rsqrt(var + eps)
    
    # Calculate base offset for this (N, C) slice
    base_offset = n * (C * spatial_size) + c * spatial_size
    
    # Process spatial block
    spatial_start = pid_spatial * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < spatial_size
    
    # Calculate actual memory offsets
    offsets = base_offset + spatial_offsets
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: BatchNorm + LeakyReLU
    norm = (x - mean) * inv_std
    bn_out = norm * gamma + beta
    out = tl.where(bn_out > 0.0, bn_out, bn_out * negative_slope)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_bn_leaky_relu(input_tensor, running_mean, running_var, weight, bias):
    """Wrapper function for the fused kernel"""
    N, C, H, W = input_tensor.shape
    output = torch.empty_like(input_tensor)
    
    spatial_size = H * W
    
    # 2D grid: (N*C) x (spatial blocks)
    grid = lambda meta: (
        N * C,
        (spatial_size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )
    
    fused_bn_leaky_relu_kernel_channelwise[grid](
        input_tensor, output,
        running_mean, running_var,
        weight, bias,
        N, C, spatial_size,
        eps=1e-05,
        negative_slope=0.01,
    )
    
    return output

def replacement_func():
    return fused_bn_leaky_relu