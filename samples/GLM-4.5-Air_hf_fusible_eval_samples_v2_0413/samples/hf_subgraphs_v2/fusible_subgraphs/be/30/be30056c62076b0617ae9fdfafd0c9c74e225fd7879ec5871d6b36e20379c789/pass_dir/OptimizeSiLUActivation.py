import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the complete forward function pattern:
    tmp_1 = torch.nn.functional.silu(in_1, inplace = True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim = 2)
    tmp_3 = split[0]
    tmp_4 = split[1] 
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)
    """
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2) 
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for the entire computation
@triton.jit
def optimized_computation_kernel(
    in_0_ptr,
    in_1_ptr,
    out_7_ptr,
    out_3_ptr, 
    out_6_ptr,
    out_4_ptr,
    batch_size,
    height,
    in_0_width,
    in_1_width_512,
    in_1_width_128,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one output element
    prog_id = tl.program_id(0)
    offset = prog_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Process tensor expansion (in_0 -> tmp_7)
    # tmp_7 has shape [1, 1, in_0_height, in_0_width]
    if offset < in_0_width:
        tl.store(out_7_ptr + offset, tl.load(in_0_ptr + offset))
    else:
        tl.store(out_7_ptr + offset, 0.0)
    
    # Process SiLU + split operations
    total_in_1_width = in_1_width_512 + in_1_width_512 + in_1_width_128
    
    # Load SiLU-processed data and handle split
    for elem_idx in range(BLOCK_SIZE):
        if prog_id * BLOCK_SIZE + elem_idx < total_in_1_width * batch_size * height:
            # Determine which split section this element belongs to
            elem_total_offset = prog_id * BLOCK_SIZE + elem_idx
            
            # Extract coordinates
            total_offset = elem_total_offset % (total_in_1_width * batch_size * height)
            batch = total_offset // (total_in_1_width * height)
            local_offset = total_offset % (total_in_1_width * height)
            height_coord = local_offset // total_in_1_width
            width_coord = local_offset % total_in_1_width
            
            # Apply SiLU and determine output
            val = tl.load(in_1_ptr + elem_total_offset)
            silu_val = val / (1.0 + tl.exp(-val))  # SiLU operation
            
            if width_coord < in_1_width_512:
                # First split (tmp_3)
                out_offset = batch * height * in_1_width_512 + height_coord * in_1_width_512 + width_coord
                tl.store(out_3_ptr + out_offset, silu_val)
            elif width_coord < in_1_width_512 * 2:
                # Second split (tmp_4)  
                out_offset = batch * height * in_1_width_512 + height_coord * in_1_width_512 + (width_coord - in_1_width_512)
                tl.store(out_4_ptr + out_offset, silu_val)
            else:
                # Third split + unsqueeze (tmp_6)
                out_offset = batch * height * in_1_width_128 + height_coord * in_1_width_128 + (width_coord - in_1_width_512 * 2)
                tl.store(out_6_ptr + out_offset, silu_val)

# Kernel wrapper
@torch.fx.wrap
def optimized_computation(in_0, in_1):
    # Get dimensions
    batch_size, height = in_1.shape[0], in_1.shape[1]
    _, in_0_width = in_0.shape
    
    in_1_width_512 = 512
    in_1_width_128 = 128
    total_in_1_width = in_1_width_512 + in_1_width_512 + in_1_width_128
    
    # Validate input
    if in_1.shape[2] != total_in_1_width:
        raise ValueError(f"Expected in_1 width {total_in_1_width}, got {in_1.shape[2]}")
    
    # Create output tensors
    # tmp_7: [1, 1, 2, 128] -> [256] elements (flattened)
    # tmp_3: [batch_size, height, 512] -> [batch_size * height * 512] elements
    # tmp_6: [batch_size, height, 1, 128] -> [batch_size * height * 128] elements (1-size dimension squeezed)  
    # tmp_4: [batch_size, height, 512] -> [batch_size * height * 512] elements
    
    total_elements = (in_0_width + batch_size * height * (in_1_width_512 + in_1_width_128 + in_1_width_512))
    
    tmp_7 = torch.empty((1, 1, 2, in_0_width), dtype=in_0.dtype, device=in_0.device)
    tmp_3 = torch.empty((batch_size, height, in_1_width_512), dtype=in_1.dtype, device=in_1.device)
    tmp_6 = torch.empty((batch_size, height, 1, in_1_width_128), dtype=in_1.dtype, device=in_1.device)
    tmp_4 = torch.empty((batch_size, height, in_1_width_512), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_computation_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_7_ptr=tmp_7,
        out_3_ptr=tmp_3,
        out_6_ptr=tmp_6, 
        out_4_ptr=tmp_4,
        batch_size=batch_size,
        height=height,
        in_0_width=in_0_width,
        in_1_width_512=in_1_width_512,
        in_1_width_128=in_1_width_128,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (tmp_7, tmp_3, tmp_6, tmp_4)

# Replacement function
def replacement_func():
    return optimized_computation