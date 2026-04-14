import torch
import triton
import triton.language as tl

def pattern(orig_in_2, orig_in_4):
    """Pattern to match slice + negate + concat + multiply operations"""
    tmp_1 = orig_in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = orig_in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * orig_in_4
    return tmp_5

def replacement_args(orig_in_2, orig_in_4):
    return (orig_in_2, orig_in_4)

@triton.jit
def fuse_slice_negate_concat_mul_kernel(
    in_2_ptr,
    in_4_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Vectorized kernel processing multiple elements per program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized computation without conditional branches
    # For elements 0-127: they come from negated second half (128-255)
    # For elements 128-255: they come from first half (0-127)
    
    # Create source indices for all elements in this block
    source_indices = offsets + tl.where(offsets < 128, 128, -128)
    
    # Load data with vectorized operations
    in_2_vals = tl.load(in_2_ptr + source_indices, mask=mask, other=0.0)
    in_4_vals = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: negate first half, keep second half, multiply with in_4
    # Use arithmetic instead of conditional for better performance
    in_2_processed = -in_2_vals * (offsets < 128) + in_2_vals * (offsets >= 128)
    
    result = in_2_processed * in_4_vals
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_slice_negate_concat_mul(in_2, in_4):
    # Optimized for small tensors [1,1,3,256] = 768 elements
    n_elements = in_2.numel()
    
    # Use smaller block size for better occupancy with small tensors
    BLOCK_SIZE = 128  # This should provide better parallelism for 768 elements
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output buffer
    out = torch.empty_like(in_2, dtype=torch.bfloat16)
    
    # Launch kernel with optimized block size
    fuse_slice_negate_concat_mul_kernel[(num_programs,)](
        in_2_ptr=in_2,
        in_4_ptr=in_4,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_slice_negate_concat_mul