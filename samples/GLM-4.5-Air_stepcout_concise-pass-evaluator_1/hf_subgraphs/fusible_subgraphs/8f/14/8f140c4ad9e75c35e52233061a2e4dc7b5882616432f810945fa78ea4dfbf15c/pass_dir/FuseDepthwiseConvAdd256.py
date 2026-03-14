import torch
import triton
import triton.language as tl

def pattern(in_6, in_5, in_4, in_7):
    # Depthwise convolution with specific parameters - matching starnet_s2.in1k (256 channels)
    tmp_6 = torch.conv2d(in_6, in_5, in_4, (1, 1), (3, 3), (1, 1), 256)
    # Element-wise addition (residual connection)
    tmp_7 = in_7 + tmp_6
    return tmp_7

def replacement_args(in_6, in_5, in_4, in_7):
    return (in_6, in_5, in_4, in_7)

@triton.jit
def depthwise_conv_add_kernel(
    input_ptr,           # [B, C, H, W] - input tensor
    weight_ptr,          # [C, 1, K, K] - depthwise convolution weights
    bias_ptr,            # [C] - bias
    residual_ptr,        # [B, C, H, W] - residual tensor
    output_ptr,          # [B, C, H, W] - output tensor
    batch_size,          # batch size
    channels,            # number of channels  
    height,              # input height
    width,               # input width
    kernel_size,         # 7
    stride_h,            # 1
    stride_w,            # 1
    pad_h,               # 3
    pad_w,               # 3
    dilation_h,          # 1
    dilation_w,          # 1
    BLOCK_SIZE_C: tl.constexpr,  # block size for channels
    BLOCK_SIZE_H: tl.constexpr,  # block size for height
    BLOCK_SIZE_W: tl.constexpr,  # block size for width
):
    # Each program handles one spatial tile for all channels in one batch
    c = tl.program_id(0)
    b = tl.program_id(1) 
    h = tl.program_id(2) * BLOCK_SIZE_H
    w = tl.program_id(3) * BLOCK_SIZE_W
    
    # Bounds checking
    channel_end = min((c + BLOCK_SIZE_C), channels)
    height_end = min((h + BLOCK_SIZE_H), height)
    width_end = min((w + BLOCK_SIZE_W), width)
    
    # Channel offset for this batch
    batch_input_ptr = input_ptr + b * channels * height * width
    batch_residual_ptr = residual_ptr + b * channels * height * width 
    batch_output_ptr = output_ptr + b * channels * height * width
    
    # Process all channels in this block
    ofc = tl.arange(0, BLOCK_SIZE_C)
    ofh = tl.arange(0, BLOCK_SIZE_H)
    ofw = tl.arange(0, BLOCK_SIZE_W)
    
    # Create 2D offset for spatial tile
    spatial_offsets = (h + ofh)[:, None] < height
    spatial_offsets2 = (w + ofw)[None, :] < width
    
    # For each channel in the block
    for c_idx in range(c, channel_end):
        # Channel offset
        channel_offset = c_idx * height * width
        
        # Load bias for this channel
        bias_val = tl.load(bias_ptr + c_idx, mask=(c_idx < channels), other=0.0)
        
        # Load weights for this channel
        weight_offset = c_idx * kernel_size * kernel_size
        weights = tl.load(weight_ptr + weight_offset, 
                         mask=tl.arange(0, kernel_size*kernel_size)[None, :] < kernel_size*kernel_size, 
                         other=0.0)
        weights = weights.reshape((kernel_size, kernel_size))
        
        # Initialize output for this channel
        channel_output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32) + bias_val
        
        # Convolution computation
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute padded coordinates
                ph_base = h + kh - pad_h
                pw_base = w + kw - pad_w
                
                if ph_base >= 0 and ph_base < height and pw_base >= 0 and pw_base < width:
                    # Load input patch
                    input_patch = tl.load(batch_input_ptr + channel_offset + 
                                        ph_base * width + pw_base + ofh[:, None] * width + ofw[None, :],
                                        mask=spatial_offsets, other=0.0)
                    
                    # Apply weights
                    weight_val = weights[kh, kw]
                    channel_output += input_patch * weight_val
        
        # Add residual connection and store
        residual_patch = tl.load(batch_residual_ptr + channel_offset + 
                               h * width + w + ofh[:, None] * width + ofw[None, :],
                               mask=spatial_offsets)
        
        output_patch = channel_output + residual_patch
        
        # Store output
        tl.store(batch_output_ptr + channel_offset + 
                h * width + w + ofh[:, None] * width + ofw[None, :],
                output_patch, mask=spatial_offsets)

@torch.fx.wrap
def fused_depthwise_conv_add_256(in_6, in_5, in_4, in_7):
    B, C, H, W = in_6.shape
    K = 7  # kernel size from the original conv2d call
    
    # Create output tensor
    output = torch.zeros_like(in_6)
    
    # Grid configuration: channels, batch, height_blocks, width_blocks
    num_channel_blocks = (C + 31) // 32  # Process 32 channels at a time
    num_height_blocks = (H + 8) // 8    # Process 8 pixels at a time
    num_width_blocks = (W + 8) // 8      # Process 8 pixels at a time
    
    # Launch kernel
    depthwise_conv_add_kernel[(num_channel_blocks, B, num_height_blocks, num_width_blocks)](
        input_ptr=in_6,
        weight_ptr=in_5,
        bias_ptr=in_4,
        residual_ptr=in_7,
        output_ptr=output,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        kernel_size=K,
        stride_h=1,
        stride_w=1,
        pad_h=3,
        pad_w=3,
        dilation_h=1,
        dilation_w=1,
        BLOCK_SIZE_C=32,  # number of channels to process in parallel
        BLOCK_SIZE_H=8,   # spatial block size for height
        BLOCK_SIZE_W=8,   # spatial block size for width
    )
    
    return output

def replacement_func():
    return fused_depthwise_conv_add_256