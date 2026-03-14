import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_0, in_3):
    # Simple pattern: multiplication and addition
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 + in_3
    return tmp_3

def replacement_args(tmp_1, tmp_0, in_3):
    return (tmp_1, tmp_0, in_3)

@triton.jit
def simple_kernel(
    tmp_1_ptr,
    tmp_0_val,
    in_3_ptr,
    out_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load tensors
    tmp_1 = tl.load(tmp_1_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Core computation: tmp_1 * tmp_0 + in_3
    result = tmp_1 * tmp_0_val + in_3
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_op(tmp_1, tmp_0, in_3):
    # Handle scalar
    if tmp_0.numel() != 1:
        raise ValueError("tmp_0 must be a scalar")
    
    in_0_val = tmp_0.item()
    num_elements = tmp_1.numel()
    
    # Use smaller block size for better stability
    block_size = 1024
    num_programs = (num_elements + block_size - 1) // block_size
    
    # Create output
    out = torch.empty_like(tmp_1)
    
    # Launch kernel
    simple_kernel[(num_programs,)](
        tmp_1_ptr=tmp_1,
        tmp_0_val=in_0_val,
        in_3_ptr=in_3,
        out_ptr=out,
        num_elements=num_elements,
        BLOCK_SIZE=block_size,
    )
    
    return out

def replacement_func():
    return simple_op