import torch
import triton
import triton.language as tl

@triton.jit
def simple_mult_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple element-wise multiplication kernel"""
    pid = tl.program_id(0)
    
    # Each program handles a block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with better memory access patterns
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    result = x_val * y_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def optimized_tensor_slice_kernel(
    key_ptr,
    sin_ptr,
    cos_ptr,
    out_final_ptr,
    out_rearranged_ptr,
    n_elements,
    half_d_model,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that handles the complex slicing and rearrangement pattern"""
    pid = tl.program_id(0)
    
    # Each program handles a block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    key_val = tl.load(key_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate offset within the model dimension (0-255)
    mod_offset = offsets % half_d_model
    
    # Apply the complex rearrangement logic:
    # Original pattern: slice key into [:128] and [128:], negate [128:], concatenate [negated_second, first]
    # Then multiply with sin and add to (key * cos)
    
    # For first half (0-127), we want to use original key values
    # For second half (128-255), we want to use negated key values
    # Then multiply this rearranged tensor with sin
    
    if mod_offset < half_d_model:
        # First half in model dimension: use original key
        rearranged_key = key_val
    else:
        # Second half in model dimension: use negated key
        rearranged_key = -key_val
    
    # Multiply with sin
    rearranged_mult = rearranged_key * sin_val
    
    # Also compute the simple key * cos for the final addition
    key_cos = key_val * cos_val
    
    # Store the rearranged multiplication result
    tl.store(out_rearranged_ptr + offsets, rearranged_mult, mask=mask)
    
    # Store the key * cos result
    tl.store(out_final_ptr + offsets, key_cos, mask=mask)

@torch.fx.wrap
def optimized_tensor_slice_pattern(key_states, cos_20, sin_20):
    """Optimized function for the complex tensor slicing and rearrangement pattern"""
    n_batch, n_heads, n_seq, d_model = key_states.shape
    half_d_model = d_model // 2
    
    # Create output tensors
    key_cos_result = torch.empty_like(key_states)  # tmp_0 equivalent
    rearranged_mult_result = torch.empty_like(key_states)  # tmp_5 equivalent
    
    BLOCK_SIZE = 256
    n_elements = key_states.numel()
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_tensor_slice_kernel[grid_size,](
        key_states,
        sin_20.expand_as(key_states),
        cos_20.expand_as(key_states),
        key_cos_result,
        rearranged_mult_result,
        n_elements,
        half_d_model,
        BLOCK_SIZE
    )
    
    # Final result: tmp_0 + tmp_5
    final_result = key_cos_result + rearranged_mult_result
    
    return key_cos_result, final_result

@triton.jit
def vectorized_mult_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized multiplication kernel with optimized memory layout"""
    pid = tl.program_id(0)
    
    # Optimized memory access patterns with vectorization
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Optimized memory coalescing
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized computation
    result = x_val * y_val
    
    # Optimized write-back
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_simple_mult(x, y):
    """Vector-optimized multiplication function"""
    out = torch.empty_like(x)
    
    # Use optimal block size for this specific tensor size
    tensor_size = x.numel()
    if tensor_size < 10000:
        BLOCK_SIZE = 128
    elif tensor_size < 100000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    n_elements = tensor_size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    vectorized_mult_kernel[grid_size,](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE
    )
    
    return out

# Simple pattern that matches a common pattern in the computation
def pattern(a, b):
    return a * b

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    return optimized_simple_mult