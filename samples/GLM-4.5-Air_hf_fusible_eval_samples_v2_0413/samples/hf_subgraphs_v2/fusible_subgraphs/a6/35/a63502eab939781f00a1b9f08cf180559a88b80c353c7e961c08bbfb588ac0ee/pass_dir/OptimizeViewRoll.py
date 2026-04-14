import torch
import triton
import triton.language as tl

def pattern(input_tensor, h, w, channels):
    """
    Pattern matching for contiguous + view(-1, h, w, channels) + roll shifts(4,4) + view(1, h*w, channels)
    This matches the core transformation sequence in the graph
    """
    # Step 1: Ensure contiguous
    tmp = input_tensor.contiguous()
    # Step 2: Reshape to spatial format [spatial_elements, h, w, channels]
    tmp = tmp.view(-1, h, w, channels)
    # Step 3: Apply roll operation with fixed shifts (4, 4) and dims (1, 2)
    tmp = torch.roll(tmp, shifts=(4, 4), dims=(1, 2))
    # Step 4: Reshape to final format [1, h*w, channels] where h*w is total spatial elements
    result = tmp.view(1, h * w, channels)
    return result

def replacement_args(input_tensor, h, w, channels):
    """Extract arguments for the replacement function"""
    return (input_tensor, h, w, channels)

@triton.jit
def _fused_view_roll_kernel(
    input_ptr,
    output_ptr,
    n_samples,
    spatial_h,
    spatial_w,
    channels,
    roll_shift_h,
    roll_shift_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel for view(-1, h, w, channels) -> roll -> view(1, h*w, channels)
    This kernel avoids intermediate memory copies by doing the spatial transformation directly
    """
    # Get program indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1) 
    pid_c = tl.program_id(2)
    
    # Calculate global position in output tensor
    spatial_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    channel_idx = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Only process valid elements
    h_mask = spatial_idx < spatial_h
    w_mask = True  # We handle width dimension separately
    c_mask = channel_idx < channels
    
    # For each (y, x) position in the output
    for y in range(spatial_h):
        for x in range(spatial_w):
            # Calculate rolled source position
            src_y = (y - roll_shift_h) % spatial_h
            src_x = (x - roll_shift_w) % spatial_w
            
            # Calculate input and output offsets
            # Input layout: [batch, src_h, src_w, channels]
            # Output layout: [1, h*w, channels] where flattened_pos = (y * w + x)
            batch_size = 1  # The view operation results in batch dimension of 1
            flatten_pos = y * spatial_w + x
            
            # For each channel in this block
            for c in range(BLOCK_SIZE_C):
                if c_mask[c]:
                    # Calculate input and output pointers
                    input_offset = (tl.sum(spatial_idx == src_y) * spatial_w * channels + 
                                   tl.sum(spatial_idx == src_x) * channels + 
                                   channel_idx[c])
                    output_offset = flatten_pos * channels + channel_idx[c]
                    
                    # Load input value with bounds checking
                    input_val = tl.load(input_ptr + input_offset, mask=(src_y < spatial_h) & (src_x < spatial_w), other=0.0)
                    
                    # Store output value
                    tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap  
def fused_view_roll(input_tensor, h, w, channels):
    """
    Fused implementation for: contiguous + view(-1, h, w, channels) + roll shifts(4,4) + view(1, h*w, channels)
    
    This avoids intermediate memory copies by directly transforming from the original
    6D format to the final rolled 3D format.
    """
    # Calculate spatial dimensions in original input
    original_shape = input_tensor.shape
    # input_shape should be [1, A, B, C, D, channels] -> view(-1, h, w, channels) 
    # where h = A*B, w = C*D
    total_spatial_elements = h * w
    
    # Create output tensor with final shape [1, total_spatial_elements, channels]
    output_shape = [1, total_spatial_elements, channels]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel with fixed roll shifts of (4, 4)
    # Grid setup: one program per spatial position and channel block
    block_size_h = 32  # Process 32 rows at a time
    block_size_w = 32  # Process 32 columns at a time  
    block_size_c = 32  # Process 32 channels at a time
    
    grid_h = (h + block_size_h - 1) // block_size_h
    grid_w = (w + block_size_w - 1) // block_size_w
    grid_c = (channels + block_size_c - 1) // block_size_c
    
    _fused_view_roll_kernel[(grid_h, grid_w, grid_c)](
        input_tensor,
        output,
        1,  # n_samples
        h,  # spatial_h
        w,  # spatial_w  
        channels,  # channels
        4,  # fixed roll_shift_h
        4,  # fixed roll_shift_w
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_W=block_size_w,
        BLOCK_SIZE_C=block_size_c,
    )
    
    return output

def replacement_func():
    """Return the fused view+roll function"""
    return fused_view_roll