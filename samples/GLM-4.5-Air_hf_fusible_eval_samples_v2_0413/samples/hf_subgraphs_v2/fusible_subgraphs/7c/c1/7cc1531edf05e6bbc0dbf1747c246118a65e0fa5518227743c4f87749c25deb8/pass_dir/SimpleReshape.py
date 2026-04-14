import torch
import triton
import triton.language as tl

def pattern(in_4):
    # Reshape operation
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    return tmp_4

def replacement_args(in_4):
    return (in_4,)

@triton.jit
def reshape_kernel(
    x_ptr,
    out_ptr,
    old_elements,
    new_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    n_programs = tl.cdiv(new_elements, BLOCK_SIZE)
    
    if pid >= n_programs:
        return
    
    # Calculate block bounds
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, new_elements)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = offsets < new_elements
    
    # For reshape, we just copy data since the layout is contiguous
    # Both [4, 128, 256] and [1, 512, 16, 16] have the same total elements (4*128*256 = 131072)
    # and are contiguous in memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store output
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def custom_reshape(x):
    # Get tensor shape
    n_elements = x.numel()
    
    # Use optimal block size
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((1, 512, 16, 16), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    reshape_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        old_elements=n_elements,
        new_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return custom_reshape