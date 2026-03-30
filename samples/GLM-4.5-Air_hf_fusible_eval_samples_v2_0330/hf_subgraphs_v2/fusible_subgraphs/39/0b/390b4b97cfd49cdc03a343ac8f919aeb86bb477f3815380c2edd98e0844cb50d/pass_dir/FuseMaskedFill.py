import torch
import triton
import triton.language as tl

# Pattern matching function - mask creation and masked fill
def pattern(in_0, tmp_2):
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, tmp_2):
    return (in_0, tmp_2)

# Optimized kernel - fused mask creation and fill
@triton.jit
def fused_mask_fill_kernel(
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
    
    # Create mask and apply in one operation
    # in_0.__eq__(0) identifies positions to fill with 1
    zero_mask = (in_0_data == 0)
    
    # Apply masked fill: where mask is True, use 1; otherwise use tmp_2_data
    out_data = tl.where(zero_mask, 1, tmp_2_data)
    
    # Store result
    tl.store(out_ptr + offsets, out_data, mask=mask)

# Wrapper function
@torch.fx.wrap
def fused_mask_fill(in_0, tmp_2):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(tmp_2)
    
    fused_mask_fill_kernel[(num_programs,)](
        in_0_ptr=in_0,
        tmp_2_ptr=tmp_2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_mask_fill