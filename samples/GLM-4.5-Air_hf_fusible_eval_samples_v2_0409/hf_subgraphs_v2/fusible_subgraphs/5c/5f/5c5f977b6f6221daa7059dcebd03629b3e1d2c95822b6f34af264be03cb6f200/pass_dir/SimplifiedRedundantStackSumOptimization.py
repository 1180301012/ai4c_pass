import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the redundant pattern: conv2d -> stack -> sum -> concat"""
    # Match the exact computation from the model - handle different input variations
    tmp_0 = in_0
    tmp_1 = in_1
    
    # Try both input variations to match different graphs
    try:
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        conv_input = in_2
        concat_input = in_3
    except:
        tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)  
        conv_input = in_3
        concat_input = in_2
    
    # Stack and sum operations (the redundant part we want to eliminate)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    
    # Final concatenation
    tmp_5 = torch.cat([tmp_4, concat_input], 1)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for replacement"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * height * width:
        return
        
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Convolution for this pixel location
    for oc in range(out_channels):
        # Load bias
        result = tl.load(bias_ptr + oc)
        
        # Weighted sum of input channels
        for ic in range(in_channels):
            # Load input value
            input_idx = (batch_idx * in_channels + ic) * height * width + h * width + w
            x_val = tl.load(input_ptr + input_idx)
            
            # Load weight value
            weight_idx = oc * in_channels + ic
            w_val = tl.load(weight_ptr + weight_idx)
            
            result += x_val * w_val
        
        # Store result
        output_idx = (batch_idx * out_channels + oc) * height * width + h * width + w
        tl.store(output_ptr + output_idx, result)

@triton.jit
def simple_conv2d_cat_kernel(
    input_ptr, weight_ptr, bias_ptr, concat_tensor_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * height * width:
        return
        
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Process both conv and concat in one kernel to avoid torch.cat
    for out_idx in range(out_channels * 2):
        if out_idx < out_channels:
            # Convolution part (first half of output)
            result = tl.load(bias_ptr + out_idx)
            
            # Weighted sum for convolution
            for ic in range(in_channels):
                input_idx = (batch_idx * in_channels + ic) * height * width + h * width + w
                x_val = tl.load(input_ptr + input_idx)
                
                weight_idx = out_idx * in_channels + ic
                w_val = tl.load(weight_ptr + weight_idx)
                
                result += x_val * w_val
            
            # Store conv result
            output_idx = (batch_idx * out_channels * 2 + out_idx) * height * width + h * width + w
            tl.store(output_ptr + output_idx, result)
        else:
            # Concatenation part (second half of output - direct copy from concat tensor)
            concat_idx = (batch_idx * out_channels + (out_idx - out_channels)) * height * width + h * width + w
            output_idx = (batch_idx * out_channels * 2 + out_idx) * height * width + h * width + w
            concat_val = tl.load(concat_tensor_ptr + concat_idx)
            tl.store(output_ptr + output_idx, concat_val)

@torch.fx.wrap  
def optimized_forward(conv_bias, conv_weight, conv_input, concat_tensor):
    """Optimized forward function that computes conv + concat in single kernel"""
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Output has 2 * out_channels from concatenation
    output_shape = (batch_size, out_channels * 2, height, width)
    output = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Flatten tensors for kernel simpler
    conv_input_flat = conv_input.reshape(-1)
    conv_weight_flat = conv_weight.reshape(-1)
    concat_tensor_flat = concat_tensor.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Launch kernel
    pixels = batch_size * height * width
    grid = (triton.cdiv(pixels, 1024),)
    
    simple_conv2d_cat_kernel[grid](
        input_ptr=conv_input_flat,
        weight_ptr=conv_weight_flat,
        bias_ptr=conv_bias,
        concat_tensor_ptr=concat_tensor_flat,
        output_ptr=output_flat,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024,
    )
    
    return output

def replacement_func():
    """Return the optimized function reference"""
    return optimized_forward