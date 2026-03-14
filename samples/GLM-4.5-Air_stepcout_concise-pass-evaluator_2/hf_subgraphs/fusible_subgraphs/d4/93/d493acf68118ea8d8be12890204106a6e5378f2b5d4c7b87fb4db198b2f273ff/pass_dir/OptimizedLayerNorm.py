import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor, weight, bias, eps=1e-05):
    # Layer normalization with weight and bias - using exact shape from model
    result = torch.nn.functional.layer_norm(input_tensor, (1024,), weight, bias, eps)
    return result

def replacement_args(input_tensor, weight, bias, eps=1e-05):
    return (input_tensor, weight, bias, eps)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of the last dimension
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data for this block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias (broadcast across the block)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / n_elements
    # Calculate variance
    x_centered = x - mean
    x2 = x_centered * x_centered
    var = tl.sum(x2, axis=0) / n_elements
    
    # Normalize and apply weight/bias
    std = tl.sqrt(var + eps)
    x_norm = (x - mean) / std
    out = x_norm * w + b
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def layernorm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    n_elements_per_norm,
    n_elements_total,
    eps: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # Multi-dimensional kernel for layer norm on [batch_size, seq_len, hidden_dim]
    pid = tl.program_id(0)
    total_norms = n_elements_total // n_elements_per_norm
    norm_idx = pid % total_norms
    
    if norm_idx >= total_norms:
        return
    
    # Calculate batch and sequence indices
    batch_idx = norm_idx // (n_elements_per_norm // 1024)  # Assuming hidden_dim=1024
    seq_idx = norm_idx % (n_elements_per_norm // 1024)
    
    # Load weight and bias (these broadcast across the hidden dimension)
    weight_offset = batch_idx * input_stride0 + seq_idx * input_stride1
    bias_offset = batch_idx * input_stride0 + seq_idx * input_stride1
    
    # Load weight and bias vectors
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_COLS), mask=tl.arange(0, BLOCK_COLS) < 1024, other=1.0)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_COLS), mask=tl.arange(0, BLOCK_COLS) < 1024, other=0.0)
    
    # Load input data for this normalization group
    x_ptrs = x_ptr + weight_offset + tl.arange(0, BLOCK_ROWS)[:, None] * input_stride2 + tl.arange(0, BLOCK_COLS)[None, :]
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_ROWS)[:, None] < BLOCK_ROWS and tl.arange(0, BLOCK_COLS)[None, :] < 1024, other=0.0)
    
    # Calculate mean and variance along the hidden dimension (axis=-1)
    mean = tl.sum(x, axis=1) / 1024
    x_centered = x - mean[:, None]
    x2 = x_centered * x_centered
    var = tl.sum(x2, axis=1) / 1024
    
    # Normalize and apply weight/bias
    std = tl.sqrt(var + eps)
    x_norm = x_centered / std[:, None]
    out = x_norm * w + b
    
    # Store result
    out_ptrs = out_ptr + weight_offset + tl.arange(0, BLOCK_ROWS)[:, None] * input_stride2 + tl.arange(0, BLOCK_COLS)[None, :]
    tl.store(out_ptrs, out, mask=tl.arange(0, BLOCK_ROWS)[:, None] < BLOCK_ROWS and tl.arange(0, BLOCK_COLS)[None, :] < 1024)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias, eps=1e-05):
    # Get tensor dimensions
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Use optimized multi-dimensional kernel
    BLOCK_ROWS = 1  # Process one row (batch x seq position) at a time
    BLOCK_COLS = 1024  # Entire hidden dimension
    
    num_norms = batch_size * seq_len  # Number of layer norm operations
    grid = (num_norms,)
    
    out = torch.empty_like(input_tensor)
    
    # Calculate strides
    input_stride0 = input_tensor.stride(0)
    input_stride1 = input_tensor.stride(1)
    input_stride2 = input_tensor.stride(2)
    
    layernorm_kernel_optimized[grid](
        input_tensor,
        weight,
        bias,
        out,
        input_stride0,
        input_stride1,
        input_stride2,
        hidden_dim,
        input_tensor.numel(),
        eps,
        BLOCK_ROWS,
        BLOCK_COLS,
    )
    
    return out

def replacement_func():
    return optimized_layernorm