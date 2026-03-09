import torch
import triton
import triton.language as tl

# Pattern matching for the first branch: negation + concat + multiply + add + type conversion
def pattern(in_6, in_5, in_2, in_4):
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4

# Argument extraction function
def replacement_args(in_6, in_5, in_2, in_4):
    return (in_6, in_5, in_2, in_4)

# Simple optimized kernel for fused element-wise operations
@triton.jit
def fused_elementwise_kernel(
    in_6_ptr, 
    in_5_ptr, 
    in_2_ptr, 
    in_4_ptr,
    out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute memory offset for each element in the block
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < total_elements
    
    # Determine half boundaries
    n3_half = total_elements // 2
    
    # Process first half (from in_6): positions 0 to n3_half-1  
    first_half_mask = (off < n3_half)
    in_6_vals = tl.load(in_6_ptr + off, mask=first_half_mask, other=0.0)
    in_2_vals = tl.load(in_2_ptr + off, mask=first_half_mask, other=0.0)
    in_4_vals = tl.load(in_4_ptr + off, mask=first_half_mask, other=0.0)
    result_first = (-in_6_vals) * in_2_vals + in_4_vals
    out_vals_first = result_first.to(tl.float32)
    tl.store(out_ptr + off, out_vals_first, mask=first_half_mask)
    
    # Process second half (from in_5): positions n3_half to total_elements-1
    second_half_mask = (off >= n3_half) & mask
    in_5_vals = tl.load(in_5_ptr + off, mask=second_half_mask, other=0.0)
    in_2_vals = tl.load(in_2_ptr + off, mask=second_half_mask, other=0.0)
    in_4_vals = tl.load(in_4_ptr + off, mask=second_half_mask, other=0.0)
    result_second = in_5_vals * in_2_vals + in_4_vals
    out_vals_second = result_second.to(tl.float32)
    tl.store(out_ptr + off, out_vals_second, mask=second_half_mask)

@torch.fx.wrap
def fused_elementwise_ops(in_6, in_5, in_2, in_4):
    # All input tensors should have the same number of elements for the computation
    # in_6 and in_5 have half the elements of in_2 and in_4 due to concatenation
    
    # Get total elements for the full concatenated tensor (in_2 and in_4)
    total_elements_full = in_2.numel()
    
    # Create output tensor with same shape as in_2 (full concatenated dimension)
    out = torch.empty_like(in_2, dtype=torch.float32, device=in_6.device)
    
    # Set up grid dimensions
    BLOCK_SIZE = 1024
    
    num_programs = (total_elements_full + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_elementwise_kernel[(num_programs,)](
        in_6_ptr=in_6,
        in_5_ptr=in_5,
        in_2_ptr=in_2,
        in_4_ptr=in_4,
        out_ptr=out,
        total_elements=total_elements_full,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_elementwise_ops