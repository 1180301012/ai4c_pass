import torch
import triton
import triton.language as tl

def pattern(tmp_12):
    """Match coordinate transformation and indexing operations."""
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 31
    tmp_14 = tmp_13
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14
    setitem = tmp_12
    tmp_14 = setitem = None
    
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 31
    tmp_17 = tmp_16
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    setitem_1 = tmp_12
    tmp_17 = setitem_1 = None
    
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 63
    tmp_20 = tmp_19
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20
    setitem_2 = tmp_12
    tmp_20 = setitem_2 = None
    
    return tmp_12

def replacement_args(tmp_12):
    return (tmp_12,)

@triton.jit
def optimized_coordinate_transform_kernel(
    coord_0_ptr,
    coord_1_ptr,
    out_0_ptr,
    out_1_ptr,
    n_elements,
    add_const_0: tl.constexpr,
    add_const_1: tl.constexpr,
    mul_const_0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for coordinate transformations (add and multiply operations)."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load coordinate values
    coord_0 = tl.load(coord_0_ptr + offsets, mask=mask, other=0)
    coord_1 = tl.load(coord_1_ptr + offsets, mask=mask, other=0)
    
    # Apply transformations: add constants and multiply first coordinate
    out_0 = (coord_0 + add_const_0) * mul_const_0
    out_1 = coord_1 + add_const_1
    
    # Store results
    tl.store(out_0_ptr + offsets, out_0, mask=mask)
    tl.store(out_1_ptr + offsets, out_1, mask=mask)

@torch.fx.wrap
def optimized_coordinate_transforms(tmp_12):
    """Optimized coordinate transformation with fused add and multiply operations."""
    n, m, _ = tmp_12.shape
    total_elements = n * m
    
    # Create output tensors
    out_0 = torch.empty((n, m), dtype=tmp_12.dtype, device=tmp_12.device)
    out_1 = torch.empty((n, m), dtype=tmp_12.dtype, device=tmp_12.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Optimize constants based on the model (these are hardcoded for the specific models we see)
    add_const_0 = 31
    add_const_1 = 31
    mul_const_0 = 63
    
    optimized_coordinate_transform_kernel[(num_programs,)](
        coord_0_ptr=tmp_12[..., 0].contiguous().data_ptr(),
        coord_1_ptr=tmp_12[..., 1].contiguous().data_ptr(),
        out_0_ptr=out_0.data_ptr(),
        out_1_ptr=out_1.data_ptr(),
        n_elements=total_elements,
        add_const_0=add_const_0,
        add_const_1=add_const_1,
        mul_const_0=mul_const_0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Combine results back into the original tensor shape using concatenation instead of stack
    result = torch.cat([out_0.unsqueeze(-1), out_1.unsqueeze(-1)], dim=-1).contiguous()
    return result

def replacement_func():
    return optimized_coordinate_transforms