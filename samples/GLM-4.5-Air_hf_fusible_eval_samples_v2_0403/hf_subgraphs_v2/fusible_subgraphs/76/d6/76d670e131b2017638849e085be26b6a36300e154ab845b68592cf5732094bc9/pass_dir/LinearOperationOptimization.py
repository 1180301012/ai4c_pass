import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    batch_size, 
    in_features, 
    out_features,
):
    """Simple linear operation kernel - one output element per program"""
    pid = tl.program_id(0)
    
    # Calculate which element we're computing: row = pid // out_features, col = pid % out_features
    row = pid // out_features
    col = pid % out_features
    
    # Early return if out of bounds
    if row >= batch_size or col >= out_features:
        return
    
    # Initialize accumulator for this output element
    result = 0.0
    
    # Compute dot product: x[row, :] @ weight[:, col] + bias[col]
    for k in range(in_features):
        # Load input element (safe access due to bounds check)
        x_elem = tl.load(x_ptr + row * in_features + k, mask=None)
        
        # Load weight element (safe access due to bounds check)
        w_elem = tl.load(weight_ptr + k * out_features + col, mask=None)
        
        # Accumulate product
        result += x_elem * w_elem
    
    # Add bias for this output feature (safe access due to bounds check)
    bias_val = tl.load(bias_ptr + col, mask=None)
    result += bias_val
    
    # Store result at position [row, col] (safe access due to bounds check)
    tl.store(out_ptr + row * out_features + col, result, mask=None)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """Optimized linear operation using Triton kernels"""
    batch_size, in_features = x.shape
    out_features = bias.shape[0]
    
    # Create output tensor as float32 for Triton computation, then convert back
    out = torch.empty((batch_size, out_features), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one program per output element
    # Total grid size: batch_size * out_features
    grid_size = (batch_size * out_features,)
    
    # Launch kernel - 1D grid as tuple
    linear_kernel[grid_size](
        x, weight, bias, out,
        batch_size, in_features, out_features
    )
    
    # Convert back to original dtype
    return out.to(x.dtype)

def pattern(x, weight, bias):
    """Match the linear operation pattern"""
    return torch.nn.functional.linear(x, weight, bias)

def replacement_args(x, weight, bias):
    """Extract arguments for the replacement"""
    return (x, weight, bias)

def replacement_func():
    """Return the optimized linear function"""
    return optimized_linear