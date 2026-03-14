import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    return tmp_2

def replacement_args(in_1, in_0):
    return (in_1, in_0, 0.1767766952966369)

@triton.jit
def fused_scalar_mult_add_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    total_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load tensors and perform fused operation
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: (in_1 * scalar) + in_0
    out_val = (in_1_val * scalar) + in_0_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_scalar_mult_add(in_1, in_0, scalar):
    # Handle the broadcasting: in_0.unsqueeze(2) broadcasts with in_1
    expanded_in_0 = in_0.unsqueeze(2)  # Add dimension at position 2
    
    # Ensure both tensors have the same shape for broadcasting
    # in_1: [1, 361, 3, 49, 49]
    # expanded_in_0: [1, 361, 1, 49, 49]
    # Result will have shape: [1, 361, 3, 49, 49]
    
    # Calculate total elements based on the broadcast shape
    total_elements = in_1.size(0) * in_1.size(1) * in_1.size(2) * in_1.size(3) * in_1.size(4)
    
    # Create output tensor with same shape as in_1
    out = torch.empty_like(in_1)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_scalar_mult_add_kernel[(num_programs,)](
        in_1,
        expanded_in_0,
        out,
        total_elements,
        scalar,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_scalar_mult_add