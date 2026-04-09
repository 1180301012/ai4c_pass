import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,          # input [N, C]
    weight_ptr,     # weight [C] 
    bias_ptr,       # bias [C]
    mean_ptr,       # output mean [N]
    var_ptr,        # output variance [N]
    out_ptr,        # output [N, C]
    N, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_offset = row_idx * C
    
    # Load data for this row
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean using Triton reduce
    row_sum = tl.sum(x, axis=0)
    row_mean = row_sum / C
    
    # Calculate variance
    x_centered = x - row_mean
    x_centered_sq = x_centered * x_centered
    row_var = tl.sum(x_centered_sq, axis=0) / C
    
    # Store mean and variance for debugging (optional)
    if mean_ptr is not None:
        tl.store(mean_ptr + row_idx, row_mean)
    if var_ptr is not None:
        tl.store(var_ptr + row_idx, row_var)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < C, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < C, other=0.0)
    
    # Layer normalization
    inv_std = 1.0 / tl.sqrt(row_var + eps)
    out = (x_centered * inv_std) * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr, 
    out_ptr,
    N, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * C
    
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized mean calculation
    row_sum = tl.sum(x, axis=0)
    row_mean = row_sum / C
    
    # Optimized variance calculation
    x_centered = x - row_mean
    x_centered_sq = x_centered * x_centered
    row_var = tl.sum(x_centered_sq, axis=0) / C
    
    # Load weight and bias vectorized
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < C, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < C, other=0.0)
    
    # Combined normalization and scaling
    inv_std = 1.0 / tl.sqrt(row_var + eps)
    out = x_centered * inv_std * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-5):
    N, C = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Choose block size based on channel dimension
    block_size = min(1024, C)
    
    # Calculate grid size (one program per row)
    grid_size = (N + block_size - 1) // block_size
    
    # Launch kernel
    optimized_layer_norm_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N,
        C=C,
        eps=eps,
        BLOCK_SIZE=block_size
    )
    
    return out

def pattern(tmp_5, in_1, in_0):
    # Match the layer normalization operation
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (tmp_5.shape[-1],), in_1, in_0, 1e-05)
    return tmp_6

def replacement_args(tmp_5, in_1, in_0):
    # Extract the channel dimension from the input tensor
    C = tmp_5.shape[-1]
    return (tmp_5, in_1, in_0, C)

@torch.fx.wrap
def optimized_layer_norm_wrapper(x, weight, bias, C):
    return optimized_layer_norm(x, weight, bias)

def replacement_func():
    return optimized_layer_norm_wrapper