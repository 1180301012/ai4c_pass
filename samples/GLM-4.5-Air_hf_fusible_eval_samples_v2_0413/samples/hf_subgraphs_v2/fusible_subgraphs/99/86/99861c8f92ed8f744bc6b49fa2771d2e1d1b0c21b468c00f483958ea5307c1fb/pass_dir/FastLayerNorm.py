import torch
import triton
import triton.language as tl
import math

def pattern(input, normalized_shape, weight, bias, eps):
    """Pattern matches layer normalization operation"""
    result = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    return result

def replacement_args(input, normalized_shape, weight, bias, eps):
    """Extract arguments for the fast layer normalization kernel"""
    return (input, weight, bias, eps)

@triton.jit
def fast_layernorm_kernel(
    input_ptr,      # Pointer to input tensor
    weight_ptr,     # Pointer to weight tensor  
    bias_ptr,       # Pointer to bias tensor
    output_ptr,     # Pointer to output tensor
    n_rows,         # Number of rows (batch_size * seq_len)
    n_features,     # Number of features (1024)
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,     # Block size for vectorization
):
    """High-performance layer normalization kernel using Triton"""
    # Each program handles one row
    row_pid = tl.program_id(0)
    
    # Check if this row is out of bounds
    if row_pid >= n_rows:
        return
    
    # Calculate start offset for this row
    row_start = row_pid * n_features
    
    # Step 1: Compute mean for this row using vectorized loads
    row_sum = 0.0
    count = 0
    
    # Process elements in blocks of BLOCK_SIZE
    for base_offset in range(0, n_features, BLOCK_SIZE):
        offset = row_start + base_offset
        end_offset = min(offset + BLOCK_SIZE, n_features)
        remaining = end_offset - offset
        
        # Vectorized load
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        
        # Load data and sum
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        block_sum = tl.sum(vals.to(tl.float32))
        row_sum += block_sum
        count += remaining
    
    # Compute mean
    row_mean = row_sum / n_features
    
    # Step 2: Compute variance for this row
    row_var = 0.0
    
    # Process elements in blocks for variance calculation
    for base_offset in range(0, n_features, BLOCK_SIZE):
        offset = row_start + base_offset
        end_offset = min(offset + BLOCK_SIZE, n_features)
        remaining = end_offset - offset
        
        # Load data
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared deviation from mean
        centered = vals - row_mean
        block_var = tl.sum(centered * centered)
        row_var += block_var
    
    # Compute variance and standard deviation
    row_var = row_var / n_features
    row_std = tl.sqrt(row_var + eps)
    
    # Step 3: Apply layer normalization and load weights/biases
    for base_offset in range(0, n_features, BLOCK_SIZE):
        offset = row_start + base_offset
        end_offset = min(offset + BLOCK_SIZE, n_features)
        
        # Load input, weights, and biases
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        
        input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Apply layer normalization: (x - mean) / std * weight + bias
        normalized = (input_vals - row_mean) / row_std
        result = normalized.to(input_vals.dtype) * weight_vals + bias_vals
        
        # Store result
        tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fast_layernorm(input, weight, bias, eps=1e-5):
    """Wrapper function to launch the fast layer normalization kernel"""
    # Get input dimensions
    batch_size = input.shape[0]
    seq_len = input.shape[1] 
    n_features = input.shape[2]
    n_rows = batch_size * seq_len
    
    # Optimized block size for vectorization
    BLOCK_SIZE = 128   # Process 128 elements per thread (power of 2 for good GPU performance)
    
    # Calculate grid dimensions (one program per row)
    grid = (n_rows, )
    
    # Allocate output tensor with same dtype as input
    output = torch.empty_like(input)
    
    # Launch the kernel
    fast_layernorm_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_rows=n_rows,
        n_features=n_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fast layer normalization function"""
    return fast_layernorm