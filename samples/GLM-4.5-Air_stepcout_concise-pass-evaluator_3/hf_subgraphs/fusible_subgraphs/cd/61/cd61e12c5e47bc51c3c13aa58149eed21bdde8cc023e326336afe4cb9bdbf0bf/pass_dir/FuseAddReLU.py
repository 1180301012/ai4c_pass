import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, tmp_2):
    """
    Match the addition + ReLU computation sequence:
    tmp_3 = in_2 + tmp_2
    tmp_4 = tmp_3.relu_()
    
    This mirrors the exact operations in model.py
    """
    tmp_3 = in_2 + tmp_2
    tmp_4 = tmp_3.relu_()
    return tmp_4

# Argument extraction function
def replacement_args(in_2, tmp_2):
    """
    Extract arguments needed for the fused add + ReLU operation
    """
    return (in_2, tmp_2)

# Optimized fused add + ReLU kernel using Triton
@triton.jit
def fused_add_relu_kernel(
    x_ptr,        # First input tensor pointer
    y_ptr,        # Second input tensor pointer  
    out_ptr,      # Output tensor pointer
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused element-wise addition and ReLU activation
    
    For each element i:
        out[i] = max(x[i] + y[i], 0.0)
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition + ReLU fusion
    # First add, then apply ReLU: max(x + y, 0)
    sum_result = x + y
    relu_result = tl.maximum(sum_result, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_add_relu(x, y):
    """
    Optimized fused element-wise addition and ReLU activation
    
    Args:
        x: First input tensor [M, N]
        y: Second input tensor [M, N] (same shape as x)
    
    Returns:
        Tensor where out[i,j] = max(x[i,j] + y[i,j], 0)
    """
    # Ensure inputs are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Check that shapes match
    if x.shape != y.shape:
        raise ValueError(f"Input shapes must match: x.shape={x.shape}, y.shape={y.shape}")
    
    # Get tensor dimensions
    n_elements = x.numel()
    
    # Optimal block size for GPU occupancy
    BLOCK_SIZE = 1024  # Good balance between latency and throughput
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch optimized kernel
    fused_add_relu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_add_relu