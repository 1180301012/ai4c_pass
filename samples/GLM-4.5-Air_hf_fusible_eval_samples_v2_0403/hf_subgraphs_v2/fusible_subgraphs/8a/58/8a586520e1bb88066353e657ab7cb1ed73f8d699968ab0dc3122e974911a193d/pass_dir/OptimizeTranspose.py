import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    tmp_4 = input_tensor.transpose(-2, -1)
    return tmp_4

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    n, c, h, w,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    # Program IDs
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    # Create offsets for channel dimension
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Create offsets for spatial dimensions
    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    # Masking
    mask_c = c_offset < c
    mask_hw = hw_offset < max(h, w)  # Handle cases where h and w may differ
    
    # Load input tensor in original layout [n, c, h, w]
    if h >= w:
        # If height >= width, transpose height and width
        input_ptrs = input_ptr + (c_offset[:, None, None] * h * w + 
                                  hw_offset[:, None] * w + 
                                  hw_offset[None, :])
        # Store in transposed layout [n, c, w, h]
        output_ptrs = output_ptr + (c_offset[:, None, None] * w * h + 
                                   hw_offset[:, None] * h + 
                                   hw_offset[None, :])
        # Create combined mask
        mask = mask_c[:, None] & (mask_hw[:, None] & (hw_offset[None, :] < w))
    else:
        # If height < width, handle transpose differently
        input_ptrs = input_ptr + (c_offset[:, None, None] * h * w + 
                                  hw_offset[None, :] * w + 
                                  hw_offset[:, None])
        # Store in transposed layout [n, c, w, h]
        output_ptrs = output_ptr + (c_offset[:, None, None] * w * h + 
                                   hw_offset[:, None] * h + 
                                   hw_offset[None, :])
        # Create combined mask
        mask = mask_c[:, None] & (hw_offset[:, None] < h) & (hw_offset[None, :] < w)
    
    # Load data
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Store transposed data
    tl.store(output_ptrs, x, mask=mask)

@torch.fx.wrap  
def optimized_transpose(input_tensor):
    n = input_tensor.shape[0]
    c = input_tensor.shape[1]
    h = input_tensor.shape[2]
    w = input_tensor.shape[3]
    
    # Determine block sizes based on tensor dimensions
    BLOCK_C = min(64, c)  # Process up to 64 channels at once
    BLOCK_HW = min(1024, max(h, w))  # Process spatial dimensions efficiently
    
    # Calculate grid dimensions
    c_tiles = (c + BLOCK_C - 1) // BLOCK_C
    hw_tiles = (max(h, w) + BLOCK_HW - 1) // BLOCK_HW
    
    # Create output tensor with transposed dimensions [n, c, w, h]
    output = torch.empty((n, c, w, h), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    transpose_kernel[(c_tiles, hw_tiles)](
        input_ptr=input_tensor,
        output_ptr=output,
        n=n, c=c, h=h, w=w,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW
    )
    
    return output

def replacement_func():
    return optimized_transpose