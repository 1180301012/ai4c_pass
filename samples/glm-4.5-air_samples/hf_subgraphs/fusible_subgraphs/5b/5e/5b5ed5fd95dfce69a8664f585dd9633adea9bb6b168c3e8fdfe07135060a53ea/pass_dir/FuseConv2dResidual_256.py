import torch
import triton
import triton.language as tl

def pattern(conv_weight, conv_bias, input_feat):
    # Conv2D with specific parameters - use literal 256 for groups during pattern matching
    conv_out = torch.conv2d(input_feat, conv_weight, conv_bias, (1, 1), (1, 1), (1, 1), 256)
    # Residual connection
    result = conv_out + input_feat
    return result

def replacement_args(conv_weight, conv_bias, input_feat):
    return (conv_weight, conv_bias, input_feat)

@triton.jit
def fused_conv2d_residual_kernel(
    input_feat_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(out_channels * batch_size * height * width, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Process with optimal block size for 256 channels
    offset = pid * BLOCK_SIZE
    remaining = BLOCK_SIZE
    vector_size = 8  # Process 8 elements at a time for better GPU utilization
    
    while remaining > 0:
        current_batch = min(vector_size, remaining)
        elem_idx = tl.arange(0, current_batch)
        output_idx = offset + elem_idx
        
        # Mask for valid elements
        mask = output_idx < out_channels * batch_size * height * width
        
        if tl.any(mask):
            # Calculate indices efficiently
            batch_idx = output_idx // (out_channels * height * width)
            channel_idx = (output_idx // (height * width)) % out_channels
            spatial_idx = output_idx % (height * width)
            h_idx = spatial_idx // width
            w_idx = spatial_idx % width
            
            # Load input features with optimized memory access
            input_feat_offset = batch_idx * in_channels * height * width + channel_idx * height * width + spatial_idx
            input_val = tl.load(input_feat_ptr + input_feat_offset, mask=mask, other=0.0)
            
            # Load bias
            bias_offset = channel_idx
            bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
            
            # Optimized 1x1 convolution for 256 channels
            # For 1x1 conv with stride=1, dilation=1, we only need the center weight
            conv_val = bias_val
            
            # Load center weight (for 1x1 convolution, only center position matters)
            weight_center_offset = channel_idx * in_channels * 9 + in_channels * 4 + 1
            weight_val = tl.load(weight_ptr + weight_center_offset, mask=mask, other=0.0)
            
            # Apply convolution (simplified for 1x1)
            conv_val = conv_val + input_val * weight_val
            
            # Apply residual connection
            result = conv_val + input_val
            
            # Store result
            output_offset = output_idx
            tl.store(output_ptr + output_offset, result, mask=mask)
        
        offset += current_batch
        remaining -= current_batch

@torch.fx.wrap
def fused_conv2d_residual(conv_weight, conv_bias, input_feat):
    batch_size, in_channels, height, width = input_feat.shape
    out_channels = conv_weight.shape[0]
    
    # Optimized block size for 256 channels and 48x48 spatial dimensions
    if width >= 48:  # 48x48 spatial dims in the 256-channel case
        BLOCK_SIZE = 4096
    elif width >= 24:
        BLOCK_SIZE = 8192
    else:
        BLOCK_SIZE = 16384
    
    total_elements = batch_size * out_channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_feat.dtype, device=input_feat.device)
    
    # Launch kernel
    fused_conv2d_residual_kernel[(num_programs,)](
        input_feat_ptr=input_feat,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_residual