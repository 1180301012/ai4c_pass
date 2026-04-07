import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern: Complete computation from 3 inputs to final flattened output
    This matches the entire forward computation:
    in_2.sigmoid() → view → expand_as → mul → add → ReLU → global_pool → flatten
    """
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3
    tmp_4 += in_0
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_full_computation_kernel(
    in_0_ptr,    # [B, C, H, W]
    in_1_ptr,    # [B, C, H, W] 
    in_2_ptr,    # [1, 1, C_in] -> [1, C, 1, 1] after sigmoid
    out_ptr,     # [B, C]
    batch,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * out_channels  # Output size
    
    # Each program handles a block of output elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Initialize result for each output element
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0.0
    
    # For each output element (batch, out_channel)
    for h_idx in range(height):
        for w_idx in range(width):
            # Calculate global offset for this spatial position and channel
            spatial_channel_offset = h_idx * width + w_idx
            output_channel = offsets % out_channels
            batch_idx = offsets // out_channels
            
            # Input indices (assuming in_1 and in_0 have shape [B, C, H, W])
            in_0_offset = batch_idx * in_channels * height * width + output_channel * height * width + h_idx * width + w_idx
            in_1_offset = batch_idx * in_channels * height * width + output_channel * height * width + h_idx * width + w_idx
            
            # Loading conditions
            in_0_valid = (in_0_offset < batch * in_channels * height * width)
            in_1_valid = (in_1_offset < batch * in_channels * height * width)
            
            # Load inputs from in_1 and apply sigmoid gate from in_2
            in_1_val = tl.load(in_1_ptr + in_1_offset, mask=in_1_valid, other=0.0)
            
            # Compute sigmoid from in_2: in_2 shape [1,1,C_in] -> we need to map C_out to C_in
            # Assuming C_out = C_in for this computation
            sigmoid_val = tl.load(in_2_ptr + output_channel, mask=output_channel < 1, other=0.0)
            sigmoid_out = 1.0 / (1.0 + tl.exp(-sigmoid_val))
            
            # Get in_0 value
            in_0_val = tl.load(in_0_ptr + in_0_offset, mask=in_0_valid, other=0.0)
            
            # Fused computation: in_1 * sigmoid_gate + in_0
            gated_in_1 = in_1_val * sigmoid_out
            combined = gated_in_1 + in_0_val
            
            # Apply ReLU and accumulate
            relu_out = tl.maximum(combined, 0.0)
            sum_val += relu_out.flatten()
            count += 1.0
    
    # Compute global average
    mean_val = sum_val / count
    
    # Store result
    tl.store(out_ptr + offsets, mean_val, mask=mask)

@torch.fx.wrap
def fused_full_computation(in_0, in_1, in_2):
    """Complete fused computation from inputs to flattened output"""
    # Get tensor shapes
    in_0_shape = in_0.shape  # [B, C, H, W]
    in_1_shape = in_1.shape  # [B, C, H, W]
    in_2_shape = in_2.shape  # [1, 1, C_in]
    
    batch, in_channels, height, width = in_0_shape
    out_channels = in_channels  # Assuming same number of channels
    
    # Create output tensor [B, C]
    out = torch.empty((batch, out_channels), dtype=in_0.dtype, device=in_0.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size based on output size
    grid_size = ((batch * out_channels) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_full_computation_kernel[grid_size](
        in_0,
        in_1, 
        in_2,
        out,
        batch,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_full_computation