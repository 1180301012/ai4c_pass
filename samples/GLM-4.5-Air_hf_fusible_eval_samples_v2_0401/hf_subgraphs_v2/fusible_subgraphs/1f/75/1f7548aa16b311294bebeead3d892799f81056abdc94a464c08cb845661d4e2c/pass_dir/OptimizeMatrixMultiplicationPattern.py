import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Match simple multiplication operation"""
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def triton_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for element-wise multiplication"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_multiply(x, y):
    """Optimized element-wise multiplication with boolean support"""
    # Determine if we're dealing with boolean tensors
    is_boolean = (x.dtype == torch.bool) or (y.dtype == torch.bool)
    
    # For boolean tensors, use native PyTorch multiplication (which is efficient on GPU)
    if is_boolean:
        # Boolean multiplication = logical AND in PyTorch
        return x * y
    
    # For numeric tensors, use the optimized Triton kernel
    # Ensure tensors are on the same device and have compatible shapes
    if x.device != y.device:
        y = y.to(x.device)
    
    # Handle broadcasting
    if x.shape != y.shape:
        # Try to broadcast to compatible shapes
        try:
            x_broadcast = x.expand_as(y) if y.numel() >= x.numel() else x
            y_broadcast = y.expand_as(x) if x.numel() >= y.numel() else y
        except:
            # If broadcasting fails, use original tensors
            x_broadcast, y_broadcast = x, y
    else:
        x_broadcast, y_broadcast = x, y
    
    N = x_broadcast.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x_broadcast)
    
    triton_multiply_kernel[(num_programs,)](
        x_ptr=x_broadcast,
        y_ptr=y_broadcast,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_multiply