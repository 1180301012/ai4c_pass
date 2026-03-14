import torch
import triton
import triton.language as tl

def pattern(in_3):
    # Match the Microsoft Swin large view-roll-view pattern
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 96, 96, 128)
    tmp_4 = torch.roll(tmp_3, shifts=(6, 6), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 9216, 128)
    return tmp_5

def replacement_args(in_3):
    return (in_3,)

# Reuse the kernel from the first pass
@triton.jit
def spatial_rolling_kernel(
    input_ptr,
    output_ptr,
    input_height,       # H dimension
    input_width,        # W dimension  
    channels,
    num_patches,        # number of patches in first dimension
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
    Optimized spatial rolling for Microsoft Swin large pattern
    """
    # Microsoft Swin large pattern: [1, 8, 12, 8, 12, 128] -> view(-1, 96, 96, 128) -> roll(6,6) -> view(1, 9216, 128)
    H, W, C = 96, 96, 128
    sequence_length = 9216
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Apply the same transformation as original to get spatial tensor
    spatial_tensor = input_tensor.view(-1, H, W, C)
    
    # Create output tensor for spatial rolling
    output_spatial = torch.empty_like(spatial_tensor)
    
    # Number of elements in spatial tensor
    num_elements = spatial_tensor.numel()
    
    # Launch kernel - use larger block size for larger tensors
    BLOCK_SIZE = 2048  # Larger block size for larger spatial dimensions
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    spatial_rolling_kernel[(num_programs,)](
        input_ptr=spatial_tensor,
        output_ptr=output_spatial,
        input_height=H,
        input_width=W,
        channels=C,
        num_patches=spatial_tensor.shape[0],
        shift_h=6,
        shift_w=6,
        num_total_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to final format
    return output_spatial.view(1, sequence_length, C)

def replacement_func():
    return optimized_spatial_rolling