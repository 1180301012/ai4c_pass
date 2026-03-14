import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Pattern: BatchNorm + ReLU fusion"""
    bn_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu_out = torch.nn.functional.relu(bn_out, inplace=True)
    return relu_out

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def batchnorm_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one channel across all spatial locations for all batch elements
    c = tl.program_id(0)
    
    if c >= C:
        return
    
    # Load statistics for this channel
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    scale = tl.load(weight_ptr + c)
    offset = tl.load(bias_ptr + c)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process all spatial locations for this channel
    spatial_size = N * H * W
    num_blocks = tl.cdiv(spatial_size, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        # Calculate offsets for this block
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Convert linear offset to (n, h, w) and then to actual memory offset
        # Memory layout: [N, C, H, W]
        n_idx = offsets // (H * W)
        hw_idx = offsets % (H * W)
        
        # Calculate actual pointer offset
        ptr_offsets = n_idx * (C * H * W) + c * (H * W) + hw_idx
        
        # Load input
        x = tl.load(x_ptr + ptr_offsets, mask=mask, other=0.0)
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (x - mean) * inv_std
        scaled = normalized * scale + offset
        
        # ReLU
        result = tl.maximum(scaled, 0.0)
        
        # Store result
        tl.store(out_ptr + ptr_offsets, result, mask=mask)

@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias):
    N, C, H, W = input_tensor.shape
    out = torch.empty_like(input_tensor)
    
    eps = 1e-05
    BLOCK_SIZE = 1024
    
    grid = (C,)
    batchnorm_relu_kernel[grid](
        input_tensor, running_mean, running_var, weight, bias, out,
        N, C, H, W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_batchnorm_relu