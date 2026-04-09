import torch
import triton
import triton.language as tl

# Pattern matching for linear transformation
def pattern(in_6, in_5, in_4):
    """
    Match the linear transformation pattern: linear = torch.nn.functional.linear(in_6, in_5, in_4)
    """
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear

# Argument extraction function
def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

# Triton kernel for optimized matrix multiplication
@triton.jit
def linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, batch_size, in_features, out_features, BLOCK_SIZE: tl.constexpr):
    """
    Optimized matrix multiplication kernel for linear transformation
    Shape: (batch_size, in_features) @ (out_features, in_features).T + (out_features,)
    """
    pid = tl.program_id(0)
    
    # Check if this program should be active
    if pid >= batch_size * out_features:
        return
    
    # Decompose linear index into batch and output dimensions
    m = pid // out_features
    n = pid % out_features
    
    # Compute output offset
    out_offset = pid
    
    # Initialize accumulator
    acc = 0.0
    
    # Main matrix multiplication loop
    for k in range(in_features):
        # Load weight (transposed layout)
        w_offset = n * in_features + k
        weight = tl.load(w_ptr + w_offset).to(tl.float32)
        
        # Load input
        x_offset = m * in_features + k
        input_val = tl.load(x_ptr + x_offset).to(tl.float32)
        
        # Multiply and accumulate
        acc += weight * input_val
    
    # Add bias
    bias = tl.load(b_ptr + n).to(tl.float32)
    result = acc + bias
    
    # Store result
    tl.store(out_ptr + out_offset, result.to(tl.float32))

# Kernel wrapper
@torch.fx.wrap
def optimized_linear(x, w, b):
    batch_size, in_features = x.shape
    out_features = w.shape[0]
    
    # Set block size for optimal performance
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = ((batch_size * out_features + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor
    out = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
    
    # Launch kernel with 1D grid
    linear_kernel[grid_size](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_linear