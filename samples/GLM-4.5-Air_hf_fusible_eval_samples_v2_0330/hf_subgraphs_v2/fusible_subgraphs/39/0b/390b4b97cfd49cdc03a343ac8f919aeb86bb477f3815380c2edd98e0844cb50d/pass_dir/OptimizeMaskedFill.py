import torch
import triton
import triton.language as tl

# Pattern matching function - just the masked fill operations
def pattern(in_0, tmp_2):
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, tmp_2):
    return (in_0, tmp_2)

# Optimized kernel - simple masked fill optimization
@triton.jit
def optimized_mask_fill_kernel(
    in_0_ptr,
    tmp_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both input tensors
    in_0_data = tl.load(in_0_ptr + offsets, mask=mask, other=0)
    tmp_2_data = tl.load(tmp_2_ptr + offsets, mask=mask, other=0)
    
    # Apply masked fill directly
    # Where in_0 == 0, use 1; otherwise use tmp_2_data
    out_data = tl.where(in_0_data == 0, 1, tmp_2_data)
    
    # Store result
    tl.store(out_ptr + offsets, out_data, mask=mask)

# Wrapper function
@torch.fx.wrap
def optimized_mask_fill(in_0, tmp_2):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(tmp_2)
    
    optimized_mask_fill_kernel[(num_programs,)](
        in_0_ptr=in_0,
        tmp_2_ptr=tmp_2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_mask_fill