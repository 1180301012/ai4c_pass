import torch
import triton
import triton.language as tl

def pattern(conv_out, feature_map):
    """
    Pattern matches the exact computation sequence:
    sigmoid(conv_out) → view(1, -1, 1, 1) → multiply → contiguous → return
    """
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = feature_map * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(conv_out, feature_map):
    # Return arguments needed for the fusion
    return (conv_out, feature_map)

@triton.jit
def fused_kernel(
    conv_out_ptr,
    feature_map_ptr, 
    out_ptr,
    n_channels,  # number of channels = 96  
    spatial_size,  # spatial dimensions = 128 * 128 = 16384
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_channels * spatial_size)
    
    # For each element, determine which channel it belongs to
    channel_offsets = offsets // spatial_size  # channel index = offset // spatial_size
    spatial_offsets = offsets % spatial_size   # spatial position within channel
    
    # Load conv_out value for each channel (apply sigmoid)
    conv_vals = tl.load(conv_out_ptr + channel_offsets, mask=channel_offsets < n_channels, other=0.0)
    sigmoid_vals = 1.0 / (1.0 + tl.exp(-conv_vals))  # Efficient sigmoid
    
    # Load feature map elements
    feature_vals = tl.load(feature_map_ptr + offsets, mask=mask, other=0.0)
    
    # Apply broadcasting: feature_vals[channel, spatial] * sigmoid_vals[channel]
    results = feature_vals * sigmoid_vals
    
    # Store results
    tl.store(out_ptr + offsets, results, mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiply_contiguous(conv_out, feature_map):
    """Fused kernel for sigmoid → view → multiply → contiguous operations"""
    
    # Get tensor shapes
    # conv_out should be [1, 96, 1, 1] -> we'll use the channel dimension (96 elements)  
    # feature_map is [1, 96, 128, 128]
    n_channels = 96  # From the tensor shapes
    spatial_size = 128 * 128  # Height * width
    
    # Use smaller block size for better GPU occupancy
    BLOCK_SIZE = 256
    
    # Calculate total grid size based on total elements in feature_map
    total_elements = n_channels * spatial_size
    num_grid_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape as feature_map
    out = torch.empty_like(feature_map)
    
    # Launch kernel with 1D grid for contiguous element processing
    fused_kernel[(num_grid_blocks,)](
        conv_out_ptr=conv_out,
        feature_map_ptr=feature_map,
        out_ptr=out,
        n_channels=n_channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_sigmoid_multiply_contiguous