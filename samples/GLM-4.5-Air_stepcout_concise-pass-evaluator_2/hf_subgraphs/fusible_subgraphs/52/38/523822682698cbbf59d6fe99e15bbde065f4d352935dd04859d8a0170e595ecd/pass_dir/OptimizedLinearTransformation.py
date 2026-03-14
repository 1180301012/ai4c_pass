import torch
import triton
import triton.language as tl

# Pattern matching function for linear transformation
def pattern(x, weight, bias):
    """Match linear transformation (matmul + bias) operation"""
    result = torch.nn.functional.linear(x, weight, bias)
    return result

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for matrix multiplication + bias (linear transformation)
@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_batch,
    n_input_features,
    n_output_features,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one output element: y[b, o] = bias[o] + sum(x[b, :] * w[o, :])
    batch_idx = tl.program_id(0)
    output_idx = tl.program_id(1)
    
    # Check bounds
    if batch_idx >= n_batch or output_idx >= n_output_features:
        return
    
    # Calculate pointers for the current batch and output
    x_row_ptr = x_ptr + batch_idx * n_input_features
    w_row_ptr = weight_ptr + output_idx * n_input_features
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Vectorized accumulation over input features
    for k in range(0, n_input_features, BLOCK_SIZE_K):
        # Load input vector
        x_ptrs = x_row_ptr + k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = k + tl.arange(0, BLOCK_SIZE_K) < n_input_features
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight vector
        w_ptrs = w_row_ptr + k + tl.arange(0, BLOCK_SIZE_K)
        w_vals = tl.load(w_ptrs, mask=x_mask, other=0.0)
        
        # Vector dot product
        dot_product = tl.sum(x_vals * w_vals)
        accumulator += dot_product
    
    # Load bias and add to accumulator
    bias_val = tl.load(bias_ptr + output_idx)
    result = accumulator + bias_val
    
    # Store result
    output_ptr = out_ptr + batch_idx * n_output_features + output_idx
    tl.store(output_ptr, result)

# Helper function to optimal block size based on input size
def get_optimal_block_sizes(batch_size, input_features, output_features):
    """Determine optimal block sizes based on tensor dimensions"""
    if batch_size <= 32 and input_features <= 256 and output_features <= 256:
        return (32, 32, 32)   # Small matrices
    elif batch_size <= 128 and input_features <= 512 and output_features <= 512:
        return (64, 64, 32)   # Medium matrices  
    else:
        return (128, 128, 64)  # Large matrices

# Kernel wrapper
@torch.fx.wrap  
def linear_fused(x, weight, bias):
    """
    Optimized linear transformation (matmul + bias) using Triton
    Handles various batch sizes and feature dimensions efficiently
    """
    batch_size = x.shape[0]
    input_features = x.shape[1]  
    output_features = weight.shape[0]
    
    # Determine optimal vectorization size based on input features
    if input_features <= 128:
        BLOCK_SIZE_K = 32
    elif input_features <= 256:
        BLOCK_SIZE_K = 64
    else:
        BLOCK_SIZE_K = 128
    
    # Calculate grid dimensions: one program per output element
    grid_batch = batch_size
    grid_output = output_features
    
    # Prepare output tensor
    out = torch.empty(batch_size, output_features, dtype=x.dtype, device=x.device)
    
    # Flatten input and weight tensors for kernel (row-major order)
    x_flat = x.view(-1)  # [batch_size * input_features]
    weight_flat = weight.view(-1)  # [output_features * input_features]
    
    # Launch kernel
    linear_kernel[(grid_batch, grid_output)](
        x_ptr=x_flat,
        weight_ptr=weight_flat,
        bias_ptr=bias,
        out_ptr=out,
        n_batch=batch_size,
        n_input_features=input_features,
        n_output_features=output_features,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

# Replacement function
def replacement_func():
    return linear_fused