"""
MixNet Fused BN + SiLU + Mean Optimization Pass

This pass fuses the following operations into a single optimized kernel:
1. Batch normalization (running mean, running var, weight, bias)
2. SiLU (Swish) activation
3. Mean pooling

The depthwise convolutions and concatenation are kept as separate operations
since they already benefit from cuDNN's highly optimized implementations.

The pattern matches the MixNet architecture's compound operations found in timm models.
"""

import torch
import triton
import triton.language as tl


# Autotune configurations for different channel sizes
@triton.autotune(
    configs=[
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256},
    ],
    key=['M', 'N'],
)
@triton.jit
def bn_silu_mean_kernel(
    # Input pointer
    x_ptr,
    # BN parameters
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Output pointers
    out_main_ptr, out_mean_ptr,
    # Strides
    stride_x, stride_out,
    # Shapes
    M, N, H, W, eps,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    - Batch normalization
    - SiLU activation
    - Mean pooling over spatial dimensions
    
    M = batch * channels
    N = spatial points (H * W)
    """
    # Calculate position
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N
    
    # Create masks
    off_m_range = off_m + tl.arange(0, BLOCK_SIZE_M)
    off_n_range = off_n + tl.arange(0, BLOCK_SIZE_N)
    mask_m = off_m_range < M
    mask_n = off_n_range < N
    
    # Load BN parameters (broadcast across spatial dimension)
    m_idx = off_m_range[:, None]
    bn_channel_idx = m_idx % N  # N is channels
    
    bn_mean = tl.load(bn_mean_ptr + bn_channel_idx, mask=mask_m[:, None], other=0.0)
    bn_var = tl.load(bn_var_ptr + bn_channel_idx, mask=mask_m[:, None], other=1.0)
    bn_weight = tl.load(bn_weight_ptr + bn_channel_idx, mask=mask_m[:, None], other=1.0)
    bn_bias = tl.load(bn_bias_ptr + bn_channel_idx, mask=mask_m[:, None], other=0.0)
    
    # Compute inv_std once
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    
    # Load input and apply BN
    # Input layout: (batch, channels, H, W)
    # Need to compute correct offsets
    batch_size = M // N  # N is channels
    channels = N
    
    # Load input data
    input_offsets = (
        (off_m_range[:, None] // N) * stride_x * H * W +  # batch offset
        (off_m_range[:, None] % N) * stride_x +  # channel offset
        (off_n_range[None, :] // W) * stride_x * W +  # H offset
        (off_n_range[None, :] % W) * stride_x  # W offset
    )
    
    x = tl.load(x_ptr + input_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Apply BN: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - bn_mean) * inv_std * bn_weight + bn_bias
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_val = tl.sigmoid(normalized)
    out = normalized * sigmoid_val
    
    # Store main output
    output_offsets = (
        (off_m_range[:, None] // N) * stride_out * H * W +
        (off_m_range[:, None] % N) * stride_out +
        (off_n_range[None, :] // W) * stride_out * W +
        (off_n_range[None, :] % W) * stride_out
    )
    tl.store(out_main_ptr + output_offsets, out, mask=mask_m[:, None] & mask_n[None, :])
    
    # Accumulate for mean pooling (atomic add across N dimension)
    # Each (batch, channel) gets contributions from all spatial positions
    mean_val = tl.sum(out, axis=1) / N
    mean_val = tl.reshape(mean_val, [BLOCK_SIZE_M, 1])
    
    # Store mean
    mean_offsets = (
        off_m_range * stride_out * 1 * 1 +
        tl.arange(0, BLOCK_SIZE_M)  # simplified
    )
    mean_store_offsets = (
        (off_m_range[:, None]) * stride_out * 1 * 1
    )
    tl.store(out_mean_ptr + mean_store_offsets, mean_val, mask=mask_m[:, None])


def pattern(in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3):
    """
    Pattern: Match the MixNet compound operation pattern.
    
    The pattern matches:
    1. Two depthwise convolutions with different kernel sizes and paddings
    2. Channel concatenation  
    3. Batch normalization
    4. SiLU activation
    5. Mean pooling
    
    Returns tmp_8 (concat result) along with outputs for proper graph matching.
    """
    # First depthwise convolution
    tmp_6 = torch.conv2d(in_8, in_4, None, (1, 1), (3, 3), (1, 1), in_4.shape[0])
    
    # Second depthwise convolution
    tmp_7 = torch.conv2d(in_9, in_5, None, (1, 1), (4, 4), (1, 1), in_5.shape[0])
    
    # Concatenate
    tmp_8 = torch.cat([in_6, in_7, tmp_6, tmp_7], 1)
    
    # Batch norm
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # SiLU activation
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    
    # Mean pooling
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    
    # Return concat result along with final outputs for proper matching
    return tmp_8, tmp_10, tmp_11


def pattern_stride2(in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3):
    """
    Pattern: Match MixNet with stride 2 convolutions.
    """
    # First depthwise convolution with stride 2
    tmp_6 = torch.conv2d(in_8, in_4, None, (2, 2), (3, 3), (1, 1), in_4.shape[0])
    
    # Second depthwise convolution with stride 2
    tmp_7 = torch.conv2d(in_9, in_5, None, (2, 2), (4, 4), (1, 1), in_5.shape[0])
    
    # Concatenate
    tmp_8 = torch.cat([in_6, in_7, tmp_6, tmp_7], 1)
    
    # Batch norm
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # SiLU activation
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    
    # Mean pooling
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    
    # Return concat result along with final outputs
    return tmp_8, tmp_10, tmp_11


@torch.fx.wrap
def bn_silu_mean_dispatch(x, bn_mean, bn_var, bn_weight, bn_bias, eps):
    """
    Dispatch function for the fused BN + SiLU + Mean kernel.
    
    Args:
        x: Input tensor of shape (batch, channels, H, W)
        bn_mean: Batch norm running mean
        bn_var: Batch norm running variance
        bn_weight: Batch norm weight (scale)
        bn_bias: Batch norm bias
        eps: Batch norm epsilon
    """
    B, C, H, W = x.shape
    
    # Allocate outputs
    out_main = torch.empty_like(x)
    out_mean = torch.empty((B, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Grid dimensions: M = B * C (treat as rows), N = H * W (treat as cols)
    M = B * C
    N = H * W
    
    # Block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = min(256, N)
    
    # Grid
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    bn_silu_mean_kernel[grid](
        x, bn_mean, bn_var, bn_weight, bn_bias,
        out_main, out_mean,
        x.stride(0), out_main.stride(0),
        M, N, H, W, eps,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
    )
    
    return out_main, out_mean


def replacement_args(in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement function.
    
    The replacement function will compute the full pattern (conv + concat + BN + silu + mean)
    using a fused kernel for the post-concat operations.
    """
    # Pass all inputs to the replacement function
    return (in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3, 1)


@torch.fx.wrap
def bn_silu_mean_full_dispatch(
    in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3, stride_val
):
    """
    Full dispatch function that computes conv + concat + fused BN + SiLU + Mean.
    
    Uses PyTorch's optimized conv2d for depthwise convolutions (cuDNN accelerated),
    and a fused Triton kernel for BN + SiLU + Mean.
    """
    # First depthwise convolution
    tmp_6 = torch.conv2d(in_8, in_4, None, (1, 1), (3, 3), (1, 1), in_4.shape[0])
    
    # Second depthwise convolution
    tmp_7 = torch.conv2d(in_9, in_5, None, (1, 1), (4, 4), (1, 1), in_5.shape[0])
    
    # Concatenate
    tmp_8 = torch.cat([in_6, in_7, tmp_6, tmp_7], 1)
    
    # Free intermediate tensors
    tmp_6 = tmp_7 = None
    
    # Apply fused BN + SiLU + Mean
    out_main, out_mean = bn_silu_mean_dispatch(tmp_8, in_0, in_1, in_3, in_2, 1e-05)
    
    return out_main, out_mean


@torch.fx.wrap
def bn_silu_mean_full_dispatch_stride2(
    in_6, in_7, in_8, in_9, in_4, in_5, in_0, in_1, in_2, in_3, stride_val
):
    """
    Full dispatch function for stride 2 variant.
    """
    # First depthwise convolution with stride 2
    tmp_6 = torch.conv2d(in_8, in_4, None, (2, 2), (3, 3), (1, 1), in_4.shape[0])
    
    # Second depthwise convolution with stride 2
    tmp_7 = torch.conv2d(in_9, in_5, None, (2, 2), (4, 4), (1, 1), in_5.shape[0])
    
    # Concatenate
    tmp_8 = torch.cat([in_6, in_7, tmp_6, tmp_7], 1)
    
    # Free intermediate tensors
    tmp_6 = tmp_7 = None
    
    # Apply fused BN + SiLU + Mean
    out_main, out_mean = bn_silu_mean_dispatch(tmp_8, in_0, in_1, in_3, in_2, 1e-05)
    
    return out_main, out_mean


def replacement_func():
    """Return the optimized replacement function."""
    return bn_silu_mean_full_dispatch