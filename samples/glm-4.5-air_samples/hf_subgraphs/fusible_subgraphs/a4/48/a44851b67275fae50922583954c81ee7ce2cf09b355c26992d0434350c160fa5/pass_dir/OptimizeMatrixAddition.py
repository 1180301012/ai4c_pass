import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple pattern that matches all tensor additions
    # Specific handling will be done in the wrapper function
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    x_batch,
    x_height,
    x_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (x_batch * x_height * x_width)
    
    # Load data with broadcasting for all dimensions
    batch = offsets // (x_height * x_width)
    h = (offsets % (x_height * x_width)) // x_width
    w = offsets % x_width
    
    x_idx = batch * x_height * x_width + h * x_width + w
    y_idx = batch * x_height * x_width + h * x_width + w
    
    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_idx, mask=mask, other=0.0)
    
    # Addition
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # Ensure tensors are on GPU for Triton
    if x.device.type != 'cuda':
        x = x.cuda()
    if y.device.type != 'cuda':
        y = y.cuda()
    
    if x.dim() == 2:
        # For 2D tensors, treat as (batch=1, height, width)
        batch_size, height, width = 1, x.shape[0], x.shape[1]
    else:
        batch_size, height, width = x.shape
    
    n_elements = batch_size * height * width
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_batch=batch_size,
        x_height=height,
        x_width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add