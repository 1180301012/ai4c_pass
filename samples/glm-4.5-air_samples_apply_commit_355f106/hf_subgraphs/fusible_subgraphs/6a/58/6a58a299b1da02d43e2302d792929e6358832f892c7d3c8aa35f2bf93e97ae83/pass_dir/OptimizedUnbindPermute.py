import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    # Pattern matches: direct indexing to avoid unbind complexity
    tmp_4 = tmp_2[:, :, 0:1, :]  # First slice along dim=2
    tmp_5 = tmp_2[:, :, 1:2, :]  # Second slice along dim=2
    return (tmp_4, tmp_5)

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def optimized_unbind_kernel(
    out1_ptr,  # First slice
    out2_ptr,  # Second slice  
    in_ptr,
    batch_size,
    dim17_size,
    dim2_size,  # Should be 2 for unbind
    dim128_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * dim17_size * dim128_size)
    
    # Load first slice [batch, 17, 0, 128]
    slice1_indices = offsets
    slice1_data = tl.load(in_ptr + slice1_indices, mask=mask, other=0.0)
    tl.store(out1_ptr + slice1_indices, slice1_data, mask=mask)
    
    # Load second slice [batch, 17, 1, 128]
    slice2_base = batch_size * dim17_size * dim128_size
    slice2_indices = slice2_base + offsets
    slice2_mask = slice2_indices < (slice2_base + batch_size * dim17_size * dim128_size)
    slice2_data = tl.load(in_ptr + slice2_indices, mask=slice2_mask, other=0.0)
    tl.store(out2_ptr + offsets, slice2_data, mask=slice2_mask)

@torch.fx.wrap
def optimized_unbind(tmp_2):
    # Input shape should be [batch, 17, 2, 128]
    batch_size, dim17, dim2, dim128 = tmp_2.shape
    
    # Output shapes: two tuples from unbind
    # tmp_4: [batch, 17, 1, 128] (first slice)
    # tmp_5: [batch, 17, 1, 128] (second slice)
    # But we need to preserve the exact dimensionality for the subsequent permute operation
    out1_shape = (batch_size, dim17, 1, dim128)
    out2_shape = (batch_size, dim17, 1, dim128)
    
    out1 = torch.empty(out1_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    out2 = torch.empty(out2_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Calculate total elements per slice and launch kernel
    elements_per_slice = batch_size * dim17 * dim128
    BLOCK_SIZE = 1024
    num_programs = (elements_per_slice + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_unbind_kernel[(num_programs,)](
        out1_ptr=out1,
        out2_ptr=out2,
        in_ptr=tmp_2,
        batch_size=batch_size,
        dim17_size=dim17,
        dim2_size=dim2,
        dim128_size=dim128,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out1, out2)

def replacement_func():
    return optimized_unbind