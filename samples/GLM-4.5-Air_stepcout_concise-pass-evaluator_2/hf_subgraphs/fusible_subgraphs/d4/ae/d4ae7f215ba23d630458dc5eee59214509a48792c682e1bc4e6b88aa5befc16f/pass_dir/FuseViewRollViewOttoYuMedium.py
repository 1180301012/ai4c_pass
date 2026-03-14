import torch
import triton
import triton.language as tl

def pattern(in_3):
    # Match the OttoYu medium view-roll-view pattern
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 56, 56, 128)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 3136, 128)
    return tmp_5

def replacement_args(in_3):
    return (in_3,)

# Use the same kernel as before but with parameters optimized for this size
@triton.jit
def spatial_rolling_kernel(
    input_ptr,
    output_ptr,
    input_height,       # H dimension
    input_width,        # W dimension  
    channels,
    num_patches,        # total patches (groups * groups)
    shift_h,
    shift_w,
    num_total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_total_elements
    
    # Calculate spatial coordinates
    patch_idx = offsets // (channels * input_height * input_width)
    h = (offsets // (channels * input_width)) % input_height
    w = (offsets // channels) % input_width
    c = offsets % channels
    
    # Apply rolling transformation
    h_rolled = (h + shift_h) % input_height
    w_rolled = (w + shift_w) % input_width
    
    # Calculate output offsets within each patch
    patch_offset = patch_idx * channels * input_height * input_width
    offset_in_patch = h_rolled * input_width * channels + w_rolled * channels + c
    output_offset = patch_offset + offset_in_patch
    
    # Load input and store output
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_spatial_rolling(input_tensor):
    """
    Optimized spatial rolling for OttoYu medium pattern
    """
    input_shape = input_tensor.shape
    
    # OttoYu medium pattern: [1, 8, 7, 8, 7, 128]
    H, W, C = 56, 56, 128
    groups = 8  # from input_shape[1] and input_shape[3]
    num_patches = groups * groups  # 8 * 8 = 64
    sequence_length = num_patches * H * W  # 64 * 56 * 56 = 200704, but should be 3136
    
    # Check if we have the correct input shape
    if len(input_shape) == 6:
        H_actual = input_shape[2] * input_shape[4]  # 7 * 7 = 49, not 56, wait this doesn't match
        # Actually, looking at the pattern: [1, 8, 7, 8, 7, 128] -> view(-1, 56, 56, 128)
        # This means they're reshaping differently. Let me use actual input shape
        H, W = 56, 56
        C = 128
        total_elements = input_tensor.numel()
        sequence_length = total_elements // C  # 1*8*7*8*7 = 3136
    else:
        # fallback
        H, W, C = 56, 56, 128
        sequence_length = 3136
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Flatten to [num_patches, H, W, C] 
    # For OttoYu medium: [1, 8, 7, 8, 7, 128] -> [8*8, 7*7, 128] = [64, 49, 128], not [64, 56, 56, 128]
    # Let me check the pattern again: they use view(-1, 56, 56, 128)
    # This suggests they're rearranging the data differently. Let me use the view transformation
    
    # Apply the same transformation as original to get spatial tensor
    spatial_tensor = input_tensor.view(-1, H, W, C)  # This should work
    
    # Create output tensor for spatial rolling
    output_spatial = torch.empty_like(spatial_tensor)
    
    # Number of elements in spatial tensor
    num_elements = spatial_tensor.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    spatial_rolling_kernel[(num_programs,)](
        input_ptr=spatial_tensor,
        output_ptr=output_spatial,
        input_height=H,
        input_width=W,
        channels=C,
        num_patches=spatial_tensor.shape[0],  # first dimension is num_patches
        shift_h=3,
        shift_w=3,
        num_total_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to final format
    return output_spatial.view(1, sequence_length, C)

def replacement_func():
    return optimized_spatial_rolling