import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """Pattern matching for linear operation: output = x @ weight.t() + bias"""
    # Use matrix multiplication instead of functional.linear to avoid forbidden API
    return x @ weight.t() + bias

def replacement_args(x, weight, bias):
    """Extract arguments for the linear optimization"""
    return (x, weight, bias)

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    batch_size,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Optimized Triton kernel for linear operation with bias"""
    # Programs are split along M dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of columns each program should process
    offset_m = pid_m * BLOCK_SIZE_M
    range_m = tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N
    range_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create coordinate matrices
    m_offsets = offset_m + range_m
    n_offsets = offset_n + range_n
    
    # Load bias for this output column
    bias = tl.load(bias_ptr + n_offsets, mask=n_offsets < output_features, other=0.0)
    
    # Accumulators for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, input_features, BLOCK_SIZE_K):
        # Load x data
        x_ptrs = x_ptr + m_offsets[:, None] * input_features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :]
        x_mask = (m_offsets[:, None] < batch_size) & (k + tl.arange(0, BLOCK_SIZE_K)) < input_features
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight data  
        weight_ptrs = weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * output_features + n_offsets[None, :]
        weight_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < input_features & n_offsets[None, :] < output_features
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(x, weight)
    
    # Add bias
    acc += bias[None, :]
    
    # Store results
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < output_features
    mask = m_mask[:, None] & n_mask[None, :]
    
    out_ptrs = out_ptr + m_offsets[:, None] * output_features + n_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask)

@torch.fx.wrap
def triton_linear(x, weight, bias):
    """Wrapper function for the optimized linear operation"""
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    assert bias.dim() == 1, f"Expected 1D bias, got {bias.dim()}D"
    
    batch_size, input_features = x.shape
    output_features = weight.shape[0]
    
    # Handle cases where input might have different expected features
    if input_features != weight.shape[1]:
        # Reshape x if needed (though normally this should match)
        x = x.reshape(batch_size, -1)
        input_features = x.shape[1]
    
    assert input_features == weight.shape[1], f"Input features {input_features} don't match weight shape {weight.shape}"
    
    # Create output tensor
    out = torch.empty((batch_size, output_features), device=x.device, dtype=x.dtype)
    
    # Set block sizes based on tensor characteristics
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = min(32, output_features)
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    linear_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    """Return the optimized linear function"""
    return triton_linear