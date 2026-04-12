import torch
import triton
import triton.language as tl

@triton.jit
def triton_split_section_kernel(
    x_ptr,
    out_ptr,
    section_start,
    section_size,
    total_elements,
    total_dim_size,
    dim2_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    if tl.any(mask):
        # Calculate position in original tensor
        offset_y = offsets // total_dim_size
        offset_x = offsets % total_dim_size
        
        # Check if this element is in our section
        section_mask = (offset_x >= section_start) & (offset_x < section_start + section_size)
        final_mask = mask & section_mask
        
        if tl.any(final_mask):
            # Load the input element
            x = tl.load(x_ptr + offsets, mask=final_mask, other=0.0)
            
            # Calculate output position: remove section offset
            output_offset = (offset_y * section_size) + (offset_x - section_start)
            
            # Store at the mapped position
            tl.store(out_ptr + output_offset, x, mask=final_mask)

@triton.jit
def triton_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    original_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = original_size * 1  # Adding a dimension
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    if tl.any(mask):
        # Calculate positions
        batch_idx = offsets // original_size
        elem_idx = offsets % original_size
        
        # Load from original tensor
        x = tl.load(x_ptr + elem_idx, mask=elem_idx < original_size, other=0.0)
        
        # Store at expanded position
        output_offset = batch_idx * (original_size * 1) + elem_idx
        tl.store(out_ptr + output_offset, x, mask=mask)

@torch.fx.wrap
def triton_split_unsqueeze(x, split_sizes, dim=2):
    """
    Custom split with unsqueeze operation using Triton kernels
    """
    size_0, size_1, size_2 = split_sizes
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    dim2_size = x.shape[dim]
    
    # Create output tensors using only allowed operations
    out1 = torch.empty([batch_size, seq_len, size_0], dtype=x.dtype, device=x.device)
    out2 = torch.empty([batch_size, seq_len, size_1], dtype=x.dtype, device=x.device)
    out3 = torch.empty([batch_size, seq_len, size_2], dtype=x.dtype, device=x.device)
    out6 = torch.empty([batch_size, seq_len, size_2, 1], dtype=x.dtype, device=x.device)
    
    total_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    total_dim_size = batch_size * seq_len
    
    # Split sections
    triton_split_section_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out1,
        section_start=0,
        section_size=size_0,
        total_elements=total_elements,
        total_dim_size=total_dim_size,
        dim2_size=dim2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    triton_split_section_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out2,
        section_start=size_0,
        section_size=size_1,
        total_elements=total_elements,
        total_dim_size=total_dim_size,
        dim2_size=dim2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    triton_split_section_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out3,
        section_start=size_0 + size_1,
        section_size=size_2,
        total_elements=total_elements,
        total_dim_size=total_dim_size,
        dim2_size=dim2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle unsqueeze for the third tensor to add a dimension
    out3_elements = out3.numel()
    unsqueeze_num_programs = (out3_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_unsqueeze_kernel[(unsqueeze_num_programs,)](
        x_ptr=out3,
        out_ptr=out6,
        original_size=out3_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2, out6

def pattern(tmp_1):
    """Match the split operation with subsequent tensor manipulations"""
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    return tmp_3, tmp_4, tmp_6

def replacement_args(tmp_1):
    """Extract arguments for the replacement"""
    return (tmp_1, [512, 512, 128])

def replacement_func():
    """Return the optimized function"""
    return triton_split_unsqueeze