import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Pattern: Conv2D + BatchNorm fused with actual model structure"""
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def fused_conv_bn_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr, output_ptr,
    batch_size, oc_channels, ih, iw, ic_channels, 
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE_OUT: tl.constexpr, BLOCK_SIZE_CH: tl.constexpr
):
    # Program ID based on output spatial dimensions and batch
    batch_idx = tl.program_id(0) // (ih * iw)
    out_h_idx = (tl.program_id(0) % (ih * iw)) // iw
    out_w_idx = tl.program_id(0) % iw
    oc_idx = tl.program_id(1)  # Output channel
    
    # Pre-process batch norm parameters 
    mean_ch = tl.load(running_mean_ptr + oc_idx)
    var_ch = tl.load(running_var_ptr + oc_idx)
    weight_ch = tl.load(bn_weight_ptr + oc_idx)
    bias_ch = tl.load(bn_bias_ptr + oc_idx)
    
    # BN scale and bias adjustment
    sqrt_var = tl.sqrt(var_ch + 1e-5)
    bn_scale = weight_ch / sqrt_var
    bn_bias_adj = bias_ch - mean_ch * bn_scale
    
    # Process multiple output positions per program
    offsets = tl.arange(0, BLOCK_SIZE_OUT)
    out_positions = (out_h_idx * iw + out_w_idx) * oc_channels + (oc_idx * BLOCK_SIZE_OUT + offsets)
    batch_offset = batch_idx * oc_channels * ih * iw
    
    # Process multiple input channels per program
    ch_offsets = tl.arange(0, BLOCK_SIZE_CH)
    
    # Accumulate result
    result = 0.0
    
    # 1x1 convolution computation
    for ic in range(0, ic_channels, BLOCK_SIZE_CH):
        # Load weight for current output channel and input channels
        if ic + ch_offsets[-1] < ic_channels:
            weight_data = tl.load(weight_ptr + 
                                (oc_idx * ic_channels + ic + ch_offsets) * 1 * 1,
                                mask=(tl.arange(0, BLOCK_SIZE_CH) < min(BLOCK_SIZE_CH, ic_channels - ic)))
        else:
            weight_data = 0.0
            
        # Load input data
        if batch_idx < batch_size and out_h_idx < ih and out_w_idx < iw:
            if ic + ch_offsets[-1] < ic_channels:
                input_offset = (batch_idx * ic_channels + ic + ch_offsets) * ih * iw + out_h_idx * iw + out_w_idx
                input_data = tl.load(input_ptr + input_offset,
                                   mask=(tl.arange(0, BLOCK_SIZE_CH) < min(BLOCK_SIZE_CH, ic_channels - ic)))
            else:
                input_data = 0.0
        else:
            input_data = 0.0
        
        # Conv2D accumulate
        result += input_data * weight_data
    
    # Apply batch normalization
    bn_result = result * bn_scale + bn_bias_adj
    
    # Store output
    output_idx = batch_offset + (oc_idx * BLOCK_SIZE_OUT + offsets) + out_h_idx * iw + (out_w_idx % BLOCK_SIZE_OUT)
    mask = (batch_idx < batch_size) & (out_h_idx < ih) & (out_w_idx < iw) & \
           (offsets < BLOCK_SIZE_OUT) & (oc_idx < oc_channels)
    
    tl.store(output_ptr + output_idx, bn_result, mask=mask)

@torch.fx.wrap
def fused_conv_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Map parameters based on pattern:
    # conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    # tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias, unused_residual = in_5, in_4, in_0, in_1, in_3, in_2, in_6
    
    # Get tensor shapes
    batch_size, ic_channels, ih, iw = conv_input.shape
    oc_channels, _, kernel_h, kernel_w = conv_weight.shape
    
    # For 1x1 conv with stride(1,1) and pad(0,0), output shape is same as input
    oh, ow = ih, iw
    
    # Create output tensor
    out = torch.empty(batch_size, oc_channels, oh, ow, dtype=x.dtype, device=x.device)
    
    # Tile sizes optimized for typical spatial and channel dimensions
    BLOCK_SIZE_OUT = 8  # Process multiple output positions per thread
    BLOCK_SIZE_CH = 256  # Process multiple channels per thread
    
    # Calculate grid dimensions
    grid_size_x = batch_size * oh * ow  # One thread per output position
    grid_size_y = oc_channels           # One program per output channel
    
    # Launch kernel
    fused_conv_bn_kernel[(grid_size_x, grid_size_y)](
        x, weight, running_mean, running_var, bn_weight, bn_bias, out,
        batch_size, oc_channels, ih, iw, ic_channels,
        1, 1, 0, 0,    # stride_h, stride_w, pad_h, pad_w
        BLOCK_SIZE_OUT, BLOCK_SIZE_CH
    )
    
    return out

def replacement_func():
    return fused_conv_bn