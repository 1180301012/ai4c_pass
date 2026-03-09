import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D + Add fusion
def pattern(in_6, in_0, in_5):
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = in_5 + tmp_5
    return tmp_6  # Only return the observable value that's used downstream

# Argument extraction function
def replacement_args(in_6, in_0, in_5):
    return (in_6, in_0, in_5)

# Simplified kernel for fused Conv2D + Add
@triton.jit
def fused_conv2d_add_kernel(
    x_ptr,          # in_6: input tensor [B, C_in, H, W]
    weight_ptr,     # in_0: weight tensor [C_out, C_in, 1, 1]
    bias_ptr,       # in_5: bias tensor [B, C_out, H, W] (for addition)
    out_ptr,        # output temporary [B, C_out, H, W]
    B, C_in, H, W, C_out,  # Tensor dimensions
    BLOCK_SIZE_C: tl.constexpr,   # Block size for channels
    BLOCK_SIZE_HW: tl.constexpr,  # Block size for spatial elements
):
    # Get program IDs
    pid_c = tl.program_id(0)  # Channel block
    pid_hw = tl.program_id(1)  # Spatial block
    
    # Calculate ranges
    c_offset = pid_c * BLOCK_SIZE_C
    hw_offset = pid_hw * BLOCK_SIZE_HW
    
    # Channel offsets for this block
    channel_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
    channel_mask = channel_offsets < C_out
    
    # Process spatial elements in this block
    hw_end = min(hw_offset + BLOCK_SIZE_HW, H * W)
    for spatial_idx in range(hw_offset, hw_end):
        # Spatial coordinates
        h = spatial_idx // W
        w = spatial_idx % W
        
        # Load input for this spatial position
        x_offset = spatial_idx * C_in
        x = tl.load(x_ptr + x_offset, mask=None)
        
        # Process each channel in the block
        for c in range(BLOCK_SIZE_C):
            channel_idx = c_offset + c
            if channel_idx < C_out:  # Only process valid channels
                
                # Load weight for this channel (1x1 convolution)
                weight_offset = channel_idx * C_in
                
                # Conv operation: dot product for this channel 
                conv_out = 0.0
                for ci in range(C_in):
                    x_val = tl.load(x_ptr + x_offset + ci, mask=None)
                    weight_val = tl.load(weight_ptr + weight_offset + ci, mask=None)
                    conv_out += x_val * weight_val
                
                # Load bias for this channel
                bias_offset = spatial_idx * C_out + channel_idx
                bias = tl.load(bias_ptr + bias_offset, mask=None)
                
                # Add bias to convolution output
                out = conv_out + bias
                
                # Store result
                out_offset = spatial_idx * C_out + channel_idx
                tl.store(out_ptr + out_offset, out, mask=None)

@torch.fx.wrap
def fused_conv2d_add(in_6, in_0, in_5):
    # Get tensor shapes
    B, C_in, H, W = in_6.shape
    C_out = in_0.shape[0]  # weight shape: [C_out, C_in, 1, 1]
    
    # Output shape should be [B, C_out, H, W]
    out = torch.empty((B, C_out, H, W), dtype=torch.float32, device=in_6.device)
    
    # Launch kernel - optimized for better performance
    BLOCK_SIZE_C = 32    # Number of channels to process per program (reduced for better performance)
    BLOCK_SIZE_HW = 128  # Number of spatial elements to process per program (reduced for better performance)
    
    # Calculate grid size
    num_programs_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_programs_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    fused_conv2d_add_kernel[(num_programs_c, num_programs_hw)](
        x_ptr=in_6,
        weight_ptr=in_0,
        bias_ptr=in_5,
        out_ptr=out,
        B=B, C_in=C_in, H=H, W=W, C_out=C_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv2d_add