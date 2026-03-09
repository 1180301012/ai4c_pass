import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: concat -> reshape -> transpose -> mul -> pad
    """
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, -1)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the optimized kernel
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_concat_mul_PAD_kernel(
    # Input tensors
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    # Output tensor  
    out_ptr,
    # Shapes
    in0_shape0, in0_shape1, in0_shape2, in0_shape3,
    in1_shape0, in1_shape1, in1_shape2, in1_shape3,
    in2_shape0, in2_shape1, in2_shape2, in2_shape3,
    in3_shape0, in3_shape1, in3_shape2, in3_shape3,
    # Constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that fuses concatenation, reshape, transpose, multiplication, and padding.
    This eliminates all CPU operations and does everything in GPU memory.
    """
    
    # Get grid coordinates
    pid = tl.program_id(0)
    
    # Calculate shapes
    target_C = in3_shape1  # Number of channels
    target_H = in3_shape2  # Height
    target_W = in3_shape3  # Width
    output_H = target_H + 1  # Add padding along dimension 2
    
    # Each program handles a portion of the spatial dimensions
    total_spatial_elements = output_H * target_W
    spatial_elements_per_program = BLOCK_SIZE_M * BLOCK_SIZE_N
    n_programs = (total_spatial_elements + spatial_elements_per_program - 1) // spatial_elements_per_program
    
    if pid >= n_programs:
        return
    
    # Calculate spatial range for this program
    spatial_start = pid * spatial_elements_per_program
    spatial_end = min((pid + 1) * spatial_elements_per_program, total_spatial_elements)
    
    # Process each spatial position
    for spatial_idx in range(spatial_start, spatial_end):
        # Convert linear spatial index to 2D coordinates
        h = spatial_idx // target_W
        w = spatial_idx % target_W
        
        # Skip if out of bounds for output (shouldn't happen due to size calculation)
        if h >= output_H:
            continue
        
        # Process all channels for this position (vectorized)
        for c in range(0, target_C, BLOCK_SIZE_M):
            # Check channel bounds
            if c >= target_C:
                continue
                
            # Calculate base offset in output tensor
            out_base_offset = (h * target_W + w) * target_C + c
            
            # For the padded row, output zeros
            if h == target_H:
                for mc in range(BLOCK_SIZE_M):
                    if c + mc < target_C:
                        out_offset = out_base_offset + mc
                        tl.store(out_ptr + out_offset, 0.0)
                continue
            
            # Calculate the linear position in the concatenated tensor
            # The concatenated tensor has shape [1, in0_shape1+in1_shape1+in2_shape1, in0_shape2, in0_shape3]
            # which equals [1, target_C, target_H, target_W] after reshape operations
            
            # This position corresponds to [c, h, w] in the final reshaped tensor
            # We need to determine where this comes from in the original concatenated data
            
            # Step 1: Calculate the linear position in the concatenated spatial dimensions
            spatial_pos = h * target_W + w
            
            # Step 2: Determine which original input tensor this position comes from
            # The original concatenation was along dim=1 (channels), so:
            # in_0 contributes [0:in0_shape1]  
            # in_1 contributes [in0_shape1:in0_shape1+in1_shape1]
            # in_2 contributes [in0_shape1+in1_shape1:total_channels]
            
            # Calculate which input tensor and the relative position within it
            total_channels_in0 = in0_shape1 * in0_shape2 * in0_shape3
            total_channels_in1 = in1_shape1 * in1_shape2 * in1_shape3
            
            # Convert the spatial position back to a channel position in the concatenated tensor
            # This is a simplified approach - we need to map properly
            concat_pos = c * (target_H * target_W) + spatial_pos
            
            # Determine which input tensor and position within it
            if concat_pos < total_channels_in0 // (target_H * target_W):
                # This comes from in_0
                # Calculate the position in in_0's spatial dimensions
                local_spatial_pos = concat_pos % (target_H * target_W)
                src_h = local_spatial_pos // target_W
                src_w = local_spatial_pos % target_W
                
                # Calculate offset in in_0 (assumes in_0 has been reshaped to have same spatial dims)
                src_ptr = in0_ptr + (src_h * target_W + src_w) * in0_shape1 + c
                cat_val = tl.load(src_ptr, other=0.0)
                
            elif concat_pos < (total_channels_in0 + total_channels_in1) // (target_H * target_W):
                # This comes from in_1  
                rel_pos = concat_pos - total_channels_in0 // (target_H * target_W)
                local_spatial_pos = rel_pos % (target_H * target_W)
                src_h = local_spatial_pos // target_W
                src_w = local_spatial_pos % target_W
                
                src_ptr = in1_ptr + (src_h * target_W + src_w) * in1_shape1 + (c - in0_shape1)
                cat_val = tl.load(src_ptr, other=0.0)
                
            else:
                # This comes from in_2
                rel_pos = concat_pos - (total_channels_in0 + total_channels_in1) // (target_H * target_W)
                local_spatial_pos = rel_pos % (target_H * target_W)
                src_h = local_spatial_pos // target_W
                src_w = local_spatial_pos % target_W
                
                src_ptr = in2_ptr + (src_h * target_W + src_w) * in2_shape1 + (c - in0_shape1 - in1_shape1)
                cat_val = tl.load(src_ptr, other=0.0)
            
            # Get value from in_3
            in3_offset = (h * target_W + w) * target_C + c
            in3_val = tl.load(in3_ptr + in3_offset, other=0.0)
            
            # Multiply and store
            result = cat_val * in3_val
            tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_concat_reshape_mul_pad(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused kernel
    """
    # Get input shapes
    in0_shape = in_0.shape
    in1_shape = in_1.shape 
    in2_shape = in_2.shape
    in3_shape = in_3.shape
    
    # Target dimensions for the multiplication (same as in_3's dimensions)
    target_C = in3_shape[1]  # Number of channels  
    target_H = in3_shape[2]  # Height (will be padded +1)
    target_W = in3_shape[3]  # Width
    
    # Create output tensor with padding (add +1 along dimension 2)
    output_H = target_H + 1  # Add 1 for padding along dimension 2
    output_shape = (1, target_C, output_H, target_W)
    out = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Get input tensor pointers - treat them as contiguous 1D arrays for the kernel
    in0_ptr = in_0.data_ptr()
    in1_ptr = in_1.data_ptr() 
    in2_ptr = in_2.data_ptr()
    in3_ptr = in_3.data_ptr()
    out_ptr = out.data_ptr()
    
    # Step 4: Launch optimized kernel that concatenates in GPU memory
    BLOCK_SIZE_M = 256  # Process M elements at once
    BLOCK_SIZE_N = 256  # Process N elements at once
    
    # Calculate grid size based on spatial dimensions
    total_spatial_elements = target_H * target_W
    spatial_elements_per_program = BLOCK_SIZE_M * BLOCK_SIZE_N
    grid_size = (total_spatial_elements + spatial_elements_per_program - 1) // spatial_elements_per_program
    
    # Launch kernel
    fused_concat_mul_PAD_kernel[grid_size](
        # Input tensor pointers
        in0_ptr, in1_ptr, in2_ptr, in3_ptr,
        # Output tensor pointer  
        out_ptr,
        # Shape information
        in0_shape[0], in0_shape[1], in0_shape[2], in0_shape[3],
        in1_shape[0], in1_shape[1], in1_shape[2], in1_shape[3],
        in2_shape[0], in2_shape[1], in2_shape[2], in2_shape[3],
        in3_shape[0], in3_shape[1], in3_shape[2], in3_shape[3],
        # Block sizes
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    """
    Return the fused kernel function
    """
    return fused_concat_reshape_mul_pad