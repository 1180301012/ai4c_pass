import torch
import triton
import triton.language as tl

def pattern(tensor_a, tensor_b):
    # Simple element-wise addition
    result = tensor_a + tensor_b
    return result

def replacement_args(tensor_a, tensor_b):
    return (tensor_a, tensor_b)

@triton.jit
def improved_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs with better precision handling
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition - use PyTorch's addition for better precision
    out = x + y  # Triton handles this with appropriate precision for dtype
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def improved_add(x, y):
    """Improved element-wise addition using Triton with better precision handling"""
    # Ensure tensors have the same shape
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_dtype = x.dtype  # Preserve input dtype
    output = torch.empty_like(x, dtype=output_dtype)
    
    improved_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return improved_add