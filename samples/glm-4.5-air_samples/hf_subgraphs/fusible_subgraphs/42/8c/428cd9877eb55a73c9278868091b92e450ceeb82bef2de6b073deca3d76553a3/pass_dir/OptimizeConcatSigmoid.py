import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    t1 = torch.cat([a, b, c], 2)
    t2 = t1.sigmoid()
    return t2

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def concat_sigmoid_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    a_size: tl.constexpr,
    b_size: tl.constexpr,
    c_size: tl.constexpr,
    total_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_size
    
    # Load elements based on which tensor section they belong to
    if offset < a_size:
        vals = tl.load(a_ptr + offset, mask=mask, other=0.0)
    elif offset < a_size + b_size:
        adjusted_offset = offset - a_size
        vals = tl.load(b_ptr + adjusted_offset, mask=mask, other=0.0)
    else:
        adjusted_offset = offset - (a_size + b_size)
        vals = tl.load(c_ptr + adjusted_offset, mask=mask, other=0.0)
    
    # Apply sigmoid directly
    result = 1.0 / (1.0 + tl.exp(-vals))
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

@torch.fx.wrap
def optimized_concat_sigmoid(a, b, c):
    # Get sizes along concatenation dimension (dim=2)
    a_size = a.size(2)
    b_size = b.size(2)
    c_size = c.size(2)
    total_size = a_size + b_size + c_size
    
    # Get other dimensions
    batch_size = a.size(0)
    feature_size = a.size(1)
    
    # Total elements in output tensor
    total_elements = batch_size * feature_size * total_size
    
    # Calculate grid and block sizes
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_shape = (batch_size, feature_size, total_size)
    output = torch.empty(output_shape, dtype=torch.float32, device=a.device)
    
    # Flatten tensors for efficient access
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    c_flat = c.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Launch kernel
    concat_sigmoid_kernel[grid_size](
        a_ptr=a_flat,
        b_ptr=b_flat,
        c_ptr=c_flat,
        output_ptr=output_flat,
        a_size=a_size,
        b_size=b_size,
        c_size=c_size,
        total_size=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_concat_sigmoid