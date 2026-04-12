import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Pattern: Complete residual block matching actual model structure"""
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_6
    tmp_7 = tmp_6
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def comprehensive_fused_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr, residual_ptr, output_ptr,
    batch_size, oc_channels, ih, iw, ic_channels,
    BLOCK_SIZE_OUT: tl.constexpr, BLOCK_SIZE_CH: tl.constexpr
):
    """Complete fused kernel for residual block: Conv2D + BatchNorm + Addition"""
    # Program ID based on output spatial dimensions and batch
    batch_idx = tl.program_id(0) // (ih * iw)
    out_h_idx = (tl.program_id(0) % (ih * iw)) // iw
    out_w_idx = tl.program_id(0) % iw
    oc_idx = tl.program_id(1)  # Output channel
    
    # Process multiple output positions per program
    offsets = tl.arange(0, BLOCK_SIZE_OUT)
    out_w_local = out_w_idx * oc_channels + (oc_idx * BLOCK_SIZE_OUT + offsets) % oc_channels
    out_positions = (out_h_idx * iw + out_w_idx) * oc_channels + out_w_local
    batch_offset = batch_idx * oc_channels * ih * iw
    
    # Process multiple input channels per program  
    ch_offsets = tl.arange(0, BLOCK_SIZE_CH)
    
    # Step 1: Compute Conv2D + BatchNorm (same as Pass 1)
    # Pre-process batch norm parameters 
    mean_ch = tl.load(running_mean_ptr + oc_idx)
    var_ch = tl.load(running_var_ptr + oc_idx)
    weight_ch = tl.load(bn_weight_ptr + oc_idx)
    bias_ch = tl.load(bn_bias_ptr + oc_idx)
    
    # BN scale and bias adjustment
    sqrt_var = tl.sqrt(var_ch + 1e-05)
    bn_scale = weight_ch / sqrt_var
    bn_bias_adj = bias_ch - mean_ch * bn_scale
    
    # Compute conv2d result
    conv_result = 0.0
    for ic in range(0, ic_channels, BLOCK_SIZE_CH):
        if ic + ch_offsets[-1] < ic_channels:
            weight_data = tl.load(weight_ptr + 
                                (oc_idx * ic_channels + ic + ch_offsets) * 1 * 1,
                                mask=(tl.arange(0, BLOCK_SIZE_CH) < min(BLOCK_SIZE_CH, ic_channels - ic)))
            
            if batch_idx < batch_size and out_h_idx < ih and out_w_idx < iw:
                input_offset = (batch_idx * ic_channels + ic + ch_offsets) * ih * iw + out_h_idx * iw + out_w_idx
                input_data = tl.load(input_ptr + input_offset,
                                   mask=(tl.arange(0, BLOCK_SIZE_CH) < min(BLOCK_SIZE_CH, ic_channels - ic)))
            else:
                input_data = 0.0
        else:
            weight_data = 0.0
            input_data = 0.0
        
        conv_result += input_data * weight_data
    
    # Apply batch normalization
    bn_result = conv_result * bn_scale + bn_bias_adj
    
    # Step 2: Add residual
    if batch_idx < batch_size and out_h_idx < ih and out_w_idx < iw:
        residual_offset = batch_idx * oc_channels * ih * iw + out_positions
        residual_data = tl.load(residual_ptr + residual_offset,
                               mask=(batch_idx < batch_size) & (out_h_idx < ih) & (out_w_idx < iw) & 
                                    (oc_idx < oc_channels) & (offsets < BLOCK_SIZE_OUT))
        final_result = bn_result + residual_data
    else:
        final_result = bn_result
    
    # Store final result
    output_idx = batch_offset + out_positions
    mask = (batch_idx < batch_size) & (out_h_idx < ih) & (out_w_idx < iw) & \
           (oc_idx < oc_channels) & (offsets < BLOCK_SIZE_OUT) & ((out_w_local % oc_channels) < oc_channels)
    
    tl.store(output_ptr + output_idx, final_result, mask=mask)

@torch.fx.wrap
def comprehensive_fused_block(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Complete fused residual block implementation with correct parameter mapping"""
    # Map parameters based on actual pattern:
    # conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    # tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # tmp_6 += in_6
    conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias, residual = in_5, in_4, in_0, in_1, in_3, in_2, in_6
    
    # Get tensor shapes
    batch_size, ic_channels, ih, iw = conv_input.shape
    oc_channels, _, kernel_h, kernel_w = conv_weight.shape
    
    # For 1x1 conv with stride(1,1) and pad(0,0), output shape matches input
    oh, ow = ih, iw
    
    # Create output tensor
    out = torch.empty(batch_size, oc_channels, oh, ow, dtype=x.dtype, device=x.device)
    
    # Optimized tile sizes
    BLOCK_SIZE_OUT = min(8, oc_channels)  # Process multiple output positions
    BLOCK_SIZE_CH = 256  # Process multiple channels
    
    # Calculate grid dimensions
    grid_size_x = batch_size * oh * ow  # One thread per output position
    grid_size_y = oc_channels           # One program per output channel
    
    # Launch comprehensive fused kernel
    comprehensive_fused_kernel[(grid_size_x, grid_size_y)](
        x, weight, running_mean, running_var, bn_weight, bn_bias, residual, out,
        batch_size, oc_channels, ih, iw, ic_channels,
        BLOCK_SIZE_OUT, BLOCK_SIZE_CH
    )
    
    return out

def replacement_func():
    return comprehensive_fused_block