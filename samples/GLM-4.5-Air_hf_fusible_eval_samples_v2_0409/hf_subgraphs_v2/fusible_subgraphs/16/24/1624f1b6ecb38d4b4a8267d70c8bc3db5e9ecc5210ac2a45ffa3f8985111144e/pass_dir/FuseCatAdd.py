import torch
import triton
import triton.language as tl

def pattern(x1, x2, x3):
    """
    Match the concatenation + addition sequence that can be fused
    tmp_10 = torch.cat((x1, x2), dim=1)
    tmp_11 = tmp_10 + x3
    """
    tmp_10 = torch.cat((x1, x2), dim=1)
    tmp_11 = tmp_10 + x3
    return tmp_11

def replacement_args(x1, x2, x3):
    return (x1, x2, x3)

@triton.jit
def fused_cat_add_kernel(
    x1_ptr, x2_ptr, x3_ptr, out_ptr,
    x1_dims0, x1_dims1, x1_dims2,
    x2_dims0, x2_dims1, x2_dims2,
    x3_dims0, x3_dims1, x3_dims2,
    out_dims0, out_dims1, out_dims2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses concatenation (dim=1) + addition operations
    """
    pid = tl.program_id(0)
    n_elements = out_dims0 * out_dims1 * out_dims2
    block_size = BLOCK_SIZE
    
    # Compute total blocks needed
    num_blocks = (n_elements + block_size - 1) // block_size
    if pid >= num_blocks:
        return
    
    # Each block handles a range of elements
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)
    offsets = tl.arange(start_idx, end_idx)
    mask = offsets < n_elements
    
    # Calculate output indices
    idx0 = offsets // (out_dims1 * out_dims2)
    remainder = offsets % (out_dims1 * out_dims2)
    idx1 = remainder // out_dims2
    idx2 = remainder % out_dims2
    
    # Determine which segment we're in for x1 and x2
    # Concatenation along dim=1 splits the dim1 dimension
    # x1 occupies [0, x1_dims1) in dim1, x2 occupies [x1_dims1, x1_dims1+x2_dims1) in dim1
    split_pos = x1_dims1
    
    if idx1 < split_pos:
        # From x1
        x1_local_idx1 = idx1
        x1_val = tl.load(x1_ptr + (idx0 * x1_dims1 * x1_dims2 + x1_local_idx1 * x1_dims2 + idx2), mask=mask)
        x2_val = 0.0  # x2 not present in this segment
    else:
        # From x2
        x1_local_idx1 = idx1 - split_pos
        x1_val = 0.0  # x1 not present in this segment
        x2_val = tl.load(x2_ptr + (idx0 * x2_dims1 * x2_dims2 + x1_local_idx1 * x2_dims2 + idx2), mask=mask)
    
    # Load value from x3 (same shape as output)
    x3_val = tl.load(x3_ptr + (idx0 * x3_dims1 * x3_dims2 + idx1 * x3_dims2 + idx2), mask=mask)
    
    # Perform fused operation: (x1 or x2) + x3
    out_val = x1_val + x2_val + x3_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_cat_add(x1, x2, x3):
    """
    Perform fused concatenation (dim=1) + addition operation
    """
    # Get input shapes
    shape1 = x1.shape
    shape2 = x2.shape
    shape3 = x3.shape
    
    # Verify shapes are compatible for concatenation along dim=1
    assert shape1[0] == shape2[0] == shape3[0], "Batch dimension mismatch"
    assert shape1[2] == shape2[2] == shape3[2], "Last dimension mismatch"
    
    # Output shape after concatenation along dim=1
    out_shape = [shape1[0], shape1[1] + shape2[1], shape1[2]]
    
    out = torch.empty(out_shape, dtype=x1.dtype, device=x1.device)
    
    BLOCK_SIZE = 1024
    n_elements = out.numel()
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_cat_add_kernel[(num_programs,)](
        x1, x2, x3, out,
        shape1[0], shape1[1], shape1[2],
        shape2[0], shape2[1], shape2[2],
        shape3[0], shape3[1], shape3[2],
        out_shape[0], out_shape[1], out_shape[2],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_cat_add