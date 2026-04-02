import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_3, in_2, other_args):
    """Match batch normalization + ReLU pattern exactly as in model"""
    # This matches: tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    #              tmp_7 = torch.nn.functional.relu(tmp_6, inplace = False)
    batch_norm_output = torch.nn.functional.batch_norm(
        tmp_5, 
        in_0, 
        in_1, 
        in_3, 
        in_2, 
        False,  # momentum
        0.1,     # eps 
        1e-05    # training flag
    )
    relu_output = torch.nn.functional.relu(batch_norm_output, inplace=False)
    return relu_output

def replacement_args(tmp_5, in_0, in_1, in_3, in_2, other_args):
    """Extract arguments for the optimized kernel"""
    return (tmp_5, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused batch normalization + ReLU kernel"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters (per channel for this block)
    channel_id = (offsets[0] // (64 * 64)) % 512  # Assuming [1, 512, 64, 64] shape
    mean = tl.load(running_mean_ptr + channel_id, mask=None)
    var = tl.load(running_var_ptr + channel_id, mask=None)
    weight_val = tl.load(weight_ptr + channel_id, mask=None)
    bias_val = tl.load(bias_ptr + channel_id, mask=None)
    
    # Apply batch normalization
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    var_plus_eps = var + eps
    sqrt_var_plus_eps = tl.sqrt(var_plus_eps)
    normalized = (x - mean) / sqrt_var_plus_eps
    batch_norm_result = normalized * weight_val + bias_val
    
    # Apply ReLU
    relu_result = tl.maximum(batch_norm_result, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, relu_result, mask=mask)

@torch.fx.wrap
def fused_batch_norm_relu(tmp_5, in_0, in_1, in_3, in_2):
    """Wrapper for fused batch normalization + ReLU"""
    # Input tensor shape: [1, 512, 64, 64]
    total_elements = tmp_5.numel()
    
    # Block size optimization for tensor cores
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tmp_5)
    
    # Launch fused kernel
    batch_norm_relu_kernel[grid_size](
        tmp_5,
        in_0,
        in_1,
        in_3,
        in_2,
        output,
        total_elements,
        1e-05,  # epsilon value
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return fused_batch_norm_relu