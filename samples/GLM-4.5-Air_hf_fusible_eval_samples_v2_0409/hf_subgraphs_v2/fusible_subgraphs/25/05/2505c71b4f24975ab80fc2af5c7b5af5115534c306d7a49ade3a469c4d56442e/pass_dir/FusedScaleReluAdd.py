import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_numel,
    y_numel,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_numel # Mask to ensure we don't go out of bounds
    
    # Load x (larger tensor)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle broadcasting: if y is scalar, broadcast it
    if y_numel == 1:
        # Load scalar once and broadcast
        y_val = tl.load(y_ptr)
        y_broadcasted = y_val
        # Calculate
        out = x + y_broadcasted
    else:
        # Load y with proper masking
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        # Calculate
        out = x + y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_numel,
    y_numel,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_numel # Mask to ensure we don't go out of bounds
    
    # Load x (larger tensor)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle broadcasting: if y is scalar, broadcast it
    if y_numel == 1:
        # Load scalar once and broadcast
        y_val = tl.load(y_ptr)
        y_broadcasted = y_val
        # Calculate
        out = x + y_broadcasted
    else:
        # Load y with proper masking
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        # Calculate
        out = x + y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    x_numel = x.numel()
    y_numel = y.numel()
    
    # Use adaptive block size based on tensor size
    if x_numel < 1024:
        BLOCK_SIZE = 256
        num_warps = 1
    elif x_numel < 10000:
        BLOCK_SIZE = 512
        num_warps = 2
    elif x_numel < 100000:
        BLOCK_SIZE = 1024
        num_warps = 4
    else:
        BLOCK_SIZE = 2048
        num_warps = 8
    
    num_programs = (x_numel + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    # Use optimized kernel with adaptive parameters
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_numel=x_numel,
        y_numel=y_numel,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out

def pattern(x, y):
    """Simple element-wise addition pattern"""
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return triton_add