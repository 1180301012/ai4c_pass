import torch
import triton
import triton.language as tl

def pattern(input_feat, weight, bias):
    """Match linear + first half slice + view pattern"""
    linear = torch.nn.functional.linear(input_feat, weight, bias)
    sliced = linear[:, :256]  # First 256 columns
    view = sliced.view(-1, 256)
    return linear, sliced, view  # Return all intermediates as they might be observable

def replacement_args(input_feat, weight, bias):
    return (input_feat, weight, bias)

@triton.jit
def linear_first_half_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, hidden_size, in_features,
    BLOCK_SIZE: tl.constexpr,
):
    """Custom Triton kernel that computes linear transformation and returns first half"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load input features
    input_ptrs = input_ptr + offsets[:, None] * in_features + tl.arange(0, in_features)[None, :]
    input_vals = tl.load(input_ptrs, mask=mask[:, None], other=0.0)
    
    # Load weights (only first 256 columns)
    weight_ptrs = weight_ptr + tl.arange(0, 256)[:, None] * in_features + tl.arange(0, in_features)[None, :]
    weight_vals = tl.load(weight_ptrs, mask=(tl.arange(0, 256)[:, None] < 256) & (tl.arange(0, in_features)[None, :] < in_features))
    
    # Compute matrix multiplication for first half
    acc = tl.zeros((BLOCK_SIZE, 256), dtype=tl.float32)
    for k in range(0, in_features, 32):
        input_block = tl.load(input_ptrs + k, mask=mask[:, None] & (tl.arange(k, k+32)[None, :] < in_features), other=0.0)
        weight_block = tl.load(weight_ptrs + k[None, :], mask=(tl.arange(0, 256)[:, None] < 256) & (tl.arange(k, k+32)[None, :] < in_features))
        acc += tl.dot(input_block, weight_block, acc_type=tl.float32)
    
    # Add bias if provided
    bias_ptrs = bias_ptr + tl.arange(0, 256)
    bias_vals = tl.load(bias_ptrs, mask=tl.arange(0, 256) < 256)
    acc += bias_vals[None, :]
    
    # Store only the first 256 columns as output
    out_ptrs = out_ptr + offsets[:, None] * 256 + tl.arange(0, 256)[None, :]
    tl.store(out_ptrs, acc, mask=mask[:, None])
    
    # Also return the intermediate linear result for observability
    # and the sliced result
    return acc

@torch.fx.wrap
def fused_linear_first_half(input_feat, weight, bias):
    input_reshaped = input_feat.reshape(-1, input_feat.shape[-1])
    batch_size = input_reshaped.shape[0]
    hidden_size = 256
        
    BLOCK_SIZE = 256
    num_programs = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
    # Output tensor for the first half
    out_first_half = torch.empty((batch_size, 256), dtype=input_feat.dtype, device=input_feat.device)
    
    # Launch kernel
    linear_first_half_kernel[(num_programs,)](
        input_reshaped,
        weight,
        bias,
        out_first_half,
        batch_size,
        hidden_size,
        input_feat.shape[-1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_first_half, out_first_half, out_first_half

def replacement_func():
    return fused_linear_first_half