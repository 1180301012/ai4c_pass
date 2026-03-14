import torch
import triton
import triton.language as tl

def pattern(relu_out, running_mean, running_var, bn_weight, bn_bias):
    # Pattern matches: batch normalization operation
    # From the models: tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    # Note: The order in models is different from standard batch_norm due to None assignments
    # Reconstructing the standard call: batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05)
    bn_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return bn_out

def replacement_args(relu_out, running_mean, running_var, bn_weight, bn_bias):
    return (relu_out, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def batch_norm_kernel(
    x_ptr,  # input tensor [N, C, H, W]
    running_mean_ptr,  # running mean [C]
    running_var_ptr,  # running var [C]
    weight_ptr,  # weight [C]
    bias_ptr,  # bias [C]
    out_ptr,  # output [N, C, H, W]
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program id
    pid = tl.program_id(0)
    
    # Calculate range for this program
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reshape to [N*C, H*W] for per-channel processing
    x_2d = x
    
    # Get channel index for each element
    linear_idx = offsets
    channel_idx = linear_idx % C
    
    # Normalize by channel (broadcast scaling factors)
    # Load normalization parameters for each channel
    idxs = channel_idx
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + idxs, mask=(idxs < C), other=0.0)
    var = tl.load(running_var_ptr + idxs, mask=(idxs < C), other=1.0)
    weight = tl.load(weight_ptr + idxs, mask=(idxs < C), other=1.0)
    bias = tl.load(bias_ptr + idxs, mask=(idxs < C), other=0.0)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x_2d - mean) * rsqrt(var + eps) * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def batch_norm_optimized(relu_out, running_mean, running_var, bn_weight, bn_bias):
    # Get tensor shape
    N, C, H, W = relu_out.shape
    numel = N * C * H * W
    
    # Allocate output tensor
    bn_out = torch.empty_like(relu_out)
    
    # Choose block size (optimize for GPU occupancy)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    batch_norm_kernel[grid](
        relu_out,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        bn_out,
        N, C, H, W,
        1e-05,  # epsilon
        BLOCK_SIZE
    )
    
    # Return optimized result
    return bn_out

def replacement_func():
    return batch_norm_optimized