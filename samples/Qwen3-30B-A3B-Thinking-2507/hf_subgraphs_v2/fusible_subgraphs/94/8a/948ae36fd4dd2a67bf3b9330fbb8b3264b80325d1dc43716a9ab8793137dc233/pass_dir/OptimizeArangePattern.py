import torch
import triton
import triton.language as tl

def pattern(stop, device):
    return torch.arange(0, stop, device=device)

def replacement_args(stop, device):
    return (stop, device)

@triton.jit
def arange_kernel(
    out_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size  # Mask to ensure we don't go out of bounds
    # Generate values: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    values = offsets
    # Store
    tl.store(out_ptr + offsets, values, mask=mask)

@torch.fx.wrap
def optimized_arange(stop, device):
    size = stop
    out = torch.empty([size], device=device, dtype=torch.int64)
    
    # Determine block size and grid size
    BLOCK_SIZE = 1024
    num_programs = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    arange_kernel[(num_programs,)](
        out_ptr=out,
        size=size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_arange