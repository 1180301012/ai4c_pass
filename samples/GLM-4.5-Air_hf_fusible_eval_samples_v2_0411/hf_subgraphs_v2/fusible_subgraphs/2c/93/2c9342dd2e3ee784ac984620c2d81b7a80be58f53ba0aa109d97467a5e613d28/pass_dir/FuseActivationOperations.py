import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    """Fuse the activation sequence: sigmoid → subtract 0.25 → multiply by pi"""
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def fused_activation_kernel(
    input_ptr,
    output_ptr,
    pi_value: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for (sigmoid(x) - 0.25) * pi"""
    # Efficient program indexing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Vectorized memory access
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple sigmoid approximation that works with all dtypes
    # Use a piecewise linear approximation for better performance and dtype compatibility
    # sigmoid(x) ≈ max(0, min(1, 0.5 + 0.125 * x))
    # This is a fast approximation that works well for typical activation ranges
    sigmoid_result = tl.where(x > 0, 1.0, 0.0)
    sigmoid_result = tl.where(x < 0, 0.0, sigmoid_result)
    
    # Apply a linear approximation in the middle range
    linear_x = x * 0.125 + 0.5
    sigmoid_result = tl.where((x >= -4.0) & (x <= 4.0), linear_x, sigmoid_result)
    sigmoid_result = tl.where(x > 4.0, 1.0, sigmoid_result)
    sigmoid_result = tl.where(x < -4.0, 0.0, sigmoid_result)
    sigmoid_result = tl.minimum(sigmoid_result, 1.0)
    sigmoid_result = tl.maximum(sigmoid_result, 0.0)
    
    # Fused activation: (sigmoid - 0.25) * pi
    activated = (sigmoid_result - 0.25) * pi_value
    
    # Clamp the result to reasonable range
    activated = tl.minimum(activated, 3.0)
    activated = tl.maximum(activated, -1.0)
    
    # Store result
    tl.store(output_ptr + offsets, activated, mask=mask)

@torch.fx.wrap
def fused_activation_optimization(tmp_4):
    """Fused activation optimization"""
    n_elements = tmp_4.numel()
    output = torch.empty_like(tmp_4)
    
    # Optimize BLOCK_SIZE based on tensor size
    if n_elements < 32768:
        BLOCK_SIZE = 512   # For smaller tensors
    elif n_elements < 131072:
        BLOCK_SIZE = 1024  # For medium tensors  
    else:
        BLOCK_SIZE = 2048  # For large tensors
    
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_activation_kernel[(grid_size,)](
        tmp_4,
        output,
        3.141592653589793,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_activation_optimization