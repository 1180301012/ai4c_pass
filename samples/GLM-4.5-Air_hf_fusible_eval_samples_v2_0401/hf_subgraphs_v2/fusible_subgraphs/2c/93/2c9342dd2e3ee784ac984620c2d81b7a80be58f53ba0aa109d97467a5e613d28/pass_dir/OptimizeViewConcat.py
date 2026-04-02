import torch
import triton
import triton.language as tl

def pattern(conv2d_output, in_3, in_4):
    """Pattern matching for view + concatenation operations"""
    tmp_3 = conv2d_output.view(conv2d_output.shape[0], 1, -1)
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    return tmp_4

def replacement_args(conv2d_output, in_3, in_4):
    return (conv2d_output, in_3, in_4)

@triton.jit
def optimized_view_concat_kernel(
    conv2d_ptr,
    in_3_ptr, 
    in_4_ptr,
    out_ptr,
    batch_size,
    in_3_dim2,
    in_4_dim2,
    conv2d_spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that combines view operation and concatenation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * (in_3_dim2 + in_4_dim2 + conv2d_spatial_size))
    
    batch_idx = offsets // (in_3_dim2 + in_4_dim2 + conv2d_spatial_size)
    spatial_idx = offsets % (in_3_dim2 + in_4_dim2 + conv2d_spatial_size)
    
    # Determine which segment we're in and load appropriate data
    output_type = tl.where(
        spatial_idx < in_3_dim2, 0,
        tl.where(spatial_idx < (in_3_dim2 + in_4_dim2), 1, 2)
    )
    
    # Load data based on segment
    if_spatial_idx = spatial_idx
    if spatial_idx >= in_3_dim2:
        if_spatial_idx = spatial_idx - in_3_dim2
    
    if spatial_idx >= (in_3_dim2 + in_4_dim2):
        if_spatial_idx = spatial_idx - (in_3_dim2 + in_4_dim2)
    
    # Load from in_3 segment
    in_3_val = tl.where(
        output_type == 0,
        tl.load(in_3_ptr + batch_idx * in_3_dim2 + if_spatial_idx, mask=mask),
        0.0
    )
    
    # Load from in_4 segment  
    in_4_val = tl.where(
        output_type == 1,
        tl.load(in_4_ptr + batch_idx * in_4_dim2 + if_spatial_idx, mask=mask),
        0.0
    )
    
    # Load from conv2d segment (after view operation)
    conv2d_val = tl.where(
        output_type == 2,
        tl.load(conv2d_ptr + batch_idx * 64 * 20 * 20 + if_spatial_idx, mask=mask),
        0.0
    )
    
    # Select appropriate value and store
    result = tl.where(output_type == 0, in_3_val,
                      tl.where(output_type == 1, in_4_val, conv2d_val))
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_view_concat(conv2d_output, in_3, in_4):
    """Optimized function that combines view and concatenation"""
    batch_size = conv2d_output.shape[0]
    in_3_dim2 = in_3.shape[2]
    in_4_dim2 = in_4.shape[2] 
    conv2d_spatial_size = conv2d_output.shape[1] * conv2d_output.shape[2] * conv2d_output.shape[3]
    
    total_size = batch_size * (in_3_dim2 + in_4_dim2 + conv2d_spatial_size)
    output_shape = (batch_size, 1, in_3_dim2 + in_4_dim2 + conv2d_spatial_size)
    
    output = torch.empty(output_shape, dtype=conv2d_output.dtype, device=conv2d_output.device)
    
    block_size = 1024
    num_programs = (total_size + block_size - 1) // block_size
    
    optimized_view_concat_kernel[(num_programs,)](
        conv2d_ptr=conv2d_output,
        in_3_ptr=in_3,
        in_4_ptr=in_4,
        out_ptr=output,
        batch_size=batch_size,
        in_3_dim2=in_3_dim2,
        in_4_dim2=in_4_dim2,
        conv2d_spatial_size=conv2d_spatial_size,
        BLOCK_SIZE=block_size,
    )
    
    return output

def replacement_func():
    return optimized_view_concat