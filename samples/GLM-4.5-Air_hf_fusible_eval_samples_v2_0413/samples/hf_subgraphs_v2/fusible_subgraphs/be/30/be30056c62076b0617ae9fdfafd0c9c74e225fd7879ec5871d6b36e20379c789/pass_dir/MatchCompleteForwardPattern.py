import torch
import triton
import triton.language as tl

# Pattern matching function - match the core computation structure
def pattern(in_0, in_1):
    """
    Match the complete forward function structure:
    This captures the essential computation flow:
    1. SiLU activation on in_1
    2. Split into [512, 512, 128] along dim=2  
    3. Take third split and unsqueeze(2)
    4. Expand in_0 and return all four results
    """
    # Apply SiLU activation (common to all patterns)
    tmp_silu = torch.nn.functional.silu(in_1, inplace=True)
    
    # Split into three parts along dimension 2
    split_parts = torch.split(tmp_silu, [512, 512, 128], dim=2)
    
    # Extract the parts
    tmp_3 = split_parts[0]  # First 512
    tmp_4 = split_parts[1]  # Second 512  
    tmp_5 = split_parts[2]  # Last 128
    
    # Apply unsqueeze to the third part
    tmp_6 = tmp_5.unsqueeze(2)
    
    # Expand in_0 (either directly or through tmp_0)
    expanded_shape = (1, 1) + in_0.shape  
    tmp_7 = in_0.view(expanded_shape)
    
    return (tmp_7, tmp_3, tmp_6, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel - completely fused computation
@triton.jit
def fused_optimized_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_7_ptr,
    out_3_ptr,
    out_6_ptr,
    out_4_ptr,
    batch_dim,
    height_dim, 
    in_0_elements,
    in_1_total_width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (in_0_elements + in_1_total_width * batch_dim * height_dim)
    
    if tl.sum(mask) > 0:
        # Process in_0 expansion (tmp_7)
        in_mask = offsets < in_0_elements
        if tl.any(in_mask):
            global_in_offset = offsets[in_mask]
            val = tl.load(in_0_ptr + global_in_offset)
            global_out_offset = global_in_offset  # Mapping [1, 1, H, W] to flattened
            tl.store(out_7_ptr + global_out_offset, val)
        
        # Process SiLU + Split operations (tmp_3, tmp_4, tmp_6)
        in_mask = offsets >= in_0_elements
        if tl.any(in_mask):
            input_offsets = offsets[in_mask] - in_0_elements
            total_in_elements = in_1_total_width * batch_dim * height_dim
            
            if tl.any(input_offsets < total_in_elements):
                valid_in_offsets = input_offsets[input_offsets < total_in_elements]
                
                # Extract coordinates
                coords = valid_in_offsets
                batch = coords // (in_1_total_width * height_dim)
                coords = coords % (in_1_total_width * height_dim)
                height = coords // in_1_total_width
                width = coords % in_1_total_width
                
                # Load values and apply SiLU
                in_values = tl.load(in_1_ptr + valid_in_offsets)
                silu_values = in_values / (1.0 + tl.exp(-in_values))
                
                # Determine output and store
                width_512 = 512
                width_128 = 128
                
                if tl.any(width < width_512):
                    first_mask = width < width_512
                    valid_coords = tl.where(first_mask)[0]
                    out_offset = batch[valid_coords] * height_dim * width_512 + height[valid_coords] * width_512 + width[valid_coords]
                    tl.store(out_3_ptr + out_offset, silu_values[valid_coords])
                
                if tl.any((width >= width_512) & (width < width_512 * 2)):
                    second_mask = (width >= width_512) & (width < width_512 * 2)
                    valid_coords = tl.where(second_mask)[0]
                    out_offset = batch[valid_coords] * height_dim * width_512 + height[valid_coords] * width_512 + (width[valid_coords] - width_512)
                    tl.store(out_4_ptr + out_offset, silu_values[valid_coords])
                
                if tl.any(width >= width_512 * 2):
                    third_mask = width >= width_512 * 2
                    valid_coords = tl.where(third_mask)[0]
                    width_coords = width[valid_coords] - width_512 * 2
                    out_offset = batch[valid_coords] * height_dim * width_128 + height[valid_coords] * width_128 + width_coords
                    tl.store(out_6_ptr + out_offset, silu_values[valid_coords])

# Kernel wrapper
@torch.fx.wrap
def fused_optimized_forward(in_0, in_1):
    batch_size, height_, in_1_total_dim = in_1.shape
    in_0_height, in_0_width = in_0.shape
    in_0_elements = in_0_height * in_0_width
    
    # Check input dimensions
    expected_in_1_dim = 512 + 512 + 128
    if in_1_total_dim != expected_in_1_dim:
        raise ValueError(f"Expected in_1 dimension {expected_in_1_dim}, got {in_1_total_dim}")
    
    # Create output tensors with correct shapes
    tmp_7 = torch.empty((1, 1, in_0_height, in_0_width), dtype=in_0.dtype, device=in_0.device)
    tmp_3 = torch.empty((batch_size, height_, 512), dtype=in_1.dtype, device=in_1.device)
    tmp_4 = torch.empty((batch_size, height_, 512), dtype=in_1.dtype, device=in_1.device)  
    tmp_6 = torch.empty((batch_size, height_, 1, 128), dtype=in_1.dtype, device=in_1.device)
    
    total_elements = in_0_elements + in_1_total_dim * batch_size * height_
    
    # Calculate launch configuration
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_optimized_kernel[grid_size](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_7_ptr=tmp_7,
        out_3_ptr=tmp_3,
        out_6_ptr=tmp_6,
        out_4_ptr=tmp_4,
        batch_dim=batch_size,
        height_dim=height_,
        in_0_elements=in_0_elements,
        in_1_total_width=in_1_total_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (tmp_7, tmp_3, tmp_6, tmp_4)

# Replacement function
def replacement_func():
    return fused_optimized_forward