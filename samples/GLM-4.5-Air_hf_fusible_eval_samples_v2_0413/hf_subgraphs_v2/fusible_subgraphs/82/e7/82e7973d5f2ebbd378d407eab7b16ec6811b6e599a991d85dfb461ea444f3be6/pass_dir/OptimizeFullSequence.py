import torch
import triton
import triton.language as tl

# Pattern matching function for the full computation sequence
def pattern(in_0, in_1):
    """Match the entire computation sequence from the models"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace = False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for the entire sequence
@triton.jit
def full_sequence_kernel(
    in_0_ptr,   # Input 0 tensor [B, C, H, W]
    in_1_ptr,   # Input 1 tensor for ReLU [B, C, H, W]
    out_ptr,    # Output tensor [B, C, 1, 1]
    batch_size,
    channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    # Calculate input offsets for this batch and channel
    batch_offset = batch_id * channels * in_height * in_width
    channel_offset = channel_id * in_height * in_width
    spatial_offset = batch_offset + channel_offset
    
    # Compute sum of all spatial elements
    spatial_sum = 0.0
    
    # Process spatial elements in blocks for better memory access
    for spatial_idx in range(0, in_height * in_width):
        offset = spatial_offset + spatial_idx
        
        # Load both inputs
        in_0_val = tl.load(in_0_ptr + offset)
        in_1_val = tl.load(in_1_ptr + offset)
        
        # ReLU + Addition: max(0, in_1_val) + in_0_val
        relu_add_val = tl.maximum(in_1_val, 0.0) + in_0_val
        
        # Accumulate sum for averaging
        spatial_sum += relu_add_val
    
    # Compute average: sum / (H * W)
    spatial_mean = spatial_sum / (in_height * in_width)
    
    # Store result at output position
    out_idx = batch_id * channels + channel_id
    tl.store(out_ptr + out_idx, spatial_mean)

@torch.fx.wrap
def optimized_full_sequence(in_0, in_1):
    """Optimized implementation of full computation sequence"""
    batch_size, channels, in_height, in_width = in_0.shape
    
    # Output shape: [batch_size, channels, 1, 1]
    out_shape = (batch_size, channels, 1, 1)
    out = torch.zeros(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Flatten output for easier 1D kernel execution
    out_flat = out.view(-1)
    
    # Launch kernel with one program per output element (batch * channels)
    total_elements = batch_size * channels
    num_programs = total_elements
    
    # Only launch if there are elements to process
    if total_elements > 0:
        full_sequence_kernel[(num_programs,)](
            in_0_ptr=in_0,
            in_1_ptr=in_1,
            out_ptr=out_flat,
            batch_size=batch_size,
            channels=channels,
            in_height=in_height,
            in_width=in_width,
            BLOCK_SIZE=1,  # Each program handles all spatial elements for one output
        )
    
    return out

# Replacement function
def replacement_func():
    return optimized_full_sequence