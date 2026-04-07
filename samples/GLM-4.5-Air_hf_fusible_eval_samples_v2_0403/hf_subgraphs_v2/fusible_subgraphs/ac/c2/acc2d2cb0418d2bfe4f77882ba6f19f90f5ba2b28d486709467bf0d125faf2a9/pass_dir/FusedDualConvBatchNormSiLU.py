import torch
import triton
import triton.language as tl
from typing import Tuple

def pattern(in_6, in_7, in_8, in_9, weight1, weight2, running_mean, running_var, weight, bias, momentum, eps):
    """
    Pattern: dual conv2d + concat + batch_norm + silu + mean
    This pattern matches the computation graph found in all provided models
    """
    # Dual convolutions - these run independently and can be fused
    out1 = torch.conv2d(in_8, weight1, None, (1, 1), (3, 3), (1, 1))
    out2 = torch.conv2d(in_9, weight2, None, (1, 1), (4, 4), (1, 1))
    
    # Concatenation of 4 tensors
    concat_out = torch.cat([in_6, in_7, out1, out2], 1)
    
    # Batch normalization
    bn_out = torch.nn.functional.batch_norm(concat_out, running_mean, running_var, weight, bias, False, momentum, eps)
    
    # SiLU activation
    silu_out = torch.nn.functional.silu(bn_out, inplace=False)
    
    # Mean reduction
    mean_out = silu_out.mean((2, 3), keepdim=True)
    
    return silu_out, mean_out

def replacement_args(in_6, in_7, in_8, in_9, weight1, weight2, running_mean, running_var, weight, bias, momentum, eps):
    return (in_6, in_7, in_8, in_9, weight1, weight2, running_mean, running_var, weight, bias, momentum, eps)

@triton.jit
def fused_dual_conv_kernel(
    # Input tensors
    in_6_ptr, in_7_ptr, in_8_ptr, in_9_ptr,
    # Weights
    weight1_ptr, weight2_ptr,
    # Batch norm params
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output
    silu_out_ptr, mean_out_ptr,
    # Tensor shapes
    batch_size, in_channels, out_channels, height, width,
    conv1_out_channels, conv2_out_channels,
    # Config
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output dimensions
    out_height = height // stride_h
    out_width = width // stride_w
    
    # Compute indices
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b = pid_b
    
    # Initialize accumulator for mean
    mean_sum = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    count = 0
    
    # Process both convolution branches in parallel
    for k1 in range(0, in_channels, BLOCK_SIZE_K):
        block_k1 = min(BLOCK_SIZE_K, in_channels - k1)
        k1_indices = k1 + tl.arange(0, block_k1)
        
        # Load input blocks for both convolution branches
        in_8_block = tl.load(in_8_ptr + b * in_channels * height * width + 
                            k1_indices[:, None, None] * height * width + 
                            (stride_h * m[:, None, None]) * width + stride_w * n[None, :, None],
                            mask=(stride_h * m[:, None, None] < height) & (stride_w * n[None, :, None] < width),
                            other=0.0)
        
        in_9_block = tl.load(in_9_ptr + b * in_channels * height * width + 
                            k1_indices[:, None, None] * height * width + 
                            (stride_h * m[:, None, None]) * width + stride_w * n[None, :, None],
                            mask=(stride_h * m[:, None, None] < height) & (stride_w * n[None, :, None] < width),
                            other=0.0)
        
        # Load weights for both convolutions
        weight1_block = tl.load(weight1_ptr + 
                               conv1_out_channels * k1_indices[:, None] + n[:, None],
                               mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < block_k1)
        
        weight2_block = tl.load(weight2_ptr + 
                               conv2_out_channels * k1_indices[:, None] + n[:, None],
                               mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < block_k1)
        
        # Convolution operations (simplified as matrix multiply for fusion)
        conv1_out = tl.sum(in_8_block[:, :, :, None] * weight1_block[None, None, :, :], axis=2)
        conv2_out = tl.sum(in_9_block[:, :, :, None] * weight2_block[None, None, :, :], axis=2)
        
        # Merge results and apply fused batch norm + silu
        fused_out = conv1_out + conv2_out
        
        # Load batch norm parameters
        bn_mean = tl.load(running_mean_ptr + n, mask=n < out_channels)
        bn_var = tl.load(running_var_ptr + n, mask=n < out_channels)
        bn_weight = tl.load(weight_ptr + n, mask=n < out_channels)
        bn_bias = tl.load(bias_ptr + n, mask=n < out_channels)
        
        # Fused batch norm + SiLU
        normalized = (fused_out - bn_mean) / tl.sqrt(bn_var + 1e-5)
        bn_out = normalized * bn_weight + bn_bias
        silu_out = bn_out / (1.0 + tl.exp(-bn_out))
        
        # Store intermediate results and accumulate for mean
        tl.store(silu_out_ptr + b * (out_channels * out_height * out_width) + 
                n * out_height * out_width + m * out_width + n,
                silu_out, mask=(m < out_height) & (n < out_width))
        
        # Accumulate for mean calculation
        mean_sum += tl.sum(silu_out, axis=(2, 3))
        count += out_height * out_width
    
    # Calculate mean
    mean_val = mean_sum / count
    tl.store(mean_out_ptr + pid_m * block_size_n + pid_n, mean_val, pid_m < out_channels // BLOCK_SIZE_M)

@torch.fx.wrap
def fused_dual_conv_bn_silu(in_6, in_7, in_8, in_9, weight1, weight2, 
                          running_mean, running_var, weight, bias, 
                          momentum, eps):
    # Get tensor shapes
    batch_size, in_channels, height, width = in_8.shape
    conv1_out_channels = weight1.shape[0]
    conv2_out_channels = weight2.shape[0]
    out_channels = in_channels + conv1_out_channels + conv2_out_channels
    
    # Output shapes
    out_height = height // 2  # Assuming stride 2
    out_width = width // 2
    
    # Create output tensors
    silu_out = torch.empty((batch_size, out_channels, out_height, out_width), 
                          dtype=in_8.dtype, device=in_8.device)
    mean_out = torch.empty((batch_size, out_channels, 1, 1), 
                          dtype=in_8.dtype, device=in_8.device)
    
    # Kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    
    # Grid configuration
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    
    # Launch kernel
    fused_dual_conv_kernel[grid_m, grid_n, grid_b](
        in_6, in_7, in_8, in_9,
        weight1, weight2,
        running_mean, running_var, weight, bias,
        silu_out, mean_out,
        batch_size, in_channels, out_channels, height, width,
        conv1_out_channels, conv2_out_channels,
        2, 2, 1, 1,  # stride_h, stride_w, pad_h, pad_w
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return silu_out, mean_out

def replacement_func():
    return fused_dual_conv_bn_silu