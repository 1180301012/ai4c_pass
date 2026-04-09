import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, mul_input):
    """
    Pattern matching for Conv2D + Sigmoid + Element-wise Multiplication + Hardtanh
    This mirrors the exact computation structure in model.py
    """
    # Conv2D operation using the exact same parameters as in model.py
    conv_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Sigmoid activation
    sigmoid_result = conv_result.sigmoid()
    
    # Element-wise multiplication
    mul_result = mul_input * sigmoid_result
    
    # Hardtanh activation
    final_result = torch.nn.functional.hardtanh(mul_result, 0.0, 6.0, False)
    
    return final_result

def replacement_args(conv_input, weight, bias, mul_input):
    """Extract arguments needed for the fused kernel"""
    return (conv_input, weight, bias, mul_input)

@triton.jit
def fused_conv_sigmoid_mul_hardtanh_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, mul_input_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    conv_stride_w: tl.constexpr, conv_stride_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Conv2D + Sigmoid + Element-wise Multiply + Hardtanh"""
    
    # Program identifiers
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # For each (batch, channel) pair, we process all height and width positions
    # Thread ID for processing multiple elements per thread
    thread_id = tl.program_id(2)
    elements_per_thread = BLOCK_SIZE
    
    # Calculate base offsets for this (batch, channel) pair
    # conv_input has shape [batch_size, in_channels, 1, 1]
    conv_input_base = batch_id * in_channels + channel_id
    
    # weight has shape [out_channels, in_channels, 1, 1]
    weight_base = channel_id * in_channels
    
    # bias has shape [out_channels]
    bias_base = channel_id
    
    # mul_input has shape [batch_size, out_channels, height, width]
    mul_input_base = (batch_id * out_channels * height * width + 
                      channel_id * height * width)
    
    # output has shape [batch_size, out_channels, height, width]
    output_base = (batch_id * out_channels * height * width + 
                   channel_id * height * width)
    
    # Process each spatial location only once with proper thread mapping
    thread_linear_id = thread_id
    
    # Calculate global 2D position for this thread
    global_pos = thread_linear_id
    height_pos = global_pos // width
    width_pos = global_pos % width
    
    # Only process if within bounds
    if height_pos < height and width_pos < width:
        # Create mask for bounds checking
        mask = (height_pos < height) & (width_pos < width)
        
        # Calculate linear offsets
        conv_input_offset = conv_input_base
        weight_offset = weight_base
        bias_offset = bias_base
        mul_input_offset = mul_input_base + height_pos * width + width_pos
        output_offset = output_base + height_pos * width + width_pos
        
        # Load values with masking
        conv_input_val = tl.load(conv_input_ptr + conv_input_offset, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
        mul_input_val = tl.load(mul_input_ptr + mul_input_offset, mask=mask, other=0.0)
        
        # Conv2D operation with 1x1 kernel (pointwise convolution)
        conv_result = conv_input_val * weight_val + bias_val
        
        # Sigmoid activation - need to handle different precisions
        # Cast to FP32 for numerical stability during sigmoid computation
        conv_result_fp32 = conv_result.to(tl.float32)
        sigmoid_result_fp32 = 1.0 / (1.0 + tl.exp(-conv_result_fp32))
        
        # Cast back to original data type
        sigmoid_result = sigmoid_result_fp32.to(conv_result.dtype)
        
        # Element-wise multiplication with mul_input
        mul_result = mul_input_val * sigmoid_result
        
        # Hardtanh activation (clamp between 0 and 6)
        final_result = tl.maximum(0.0, tl.minimum(mul_result, 6.0))
        
        # Store result
        tl.store(output_ptr + output_offset, final_result, mask=mask)



@torch.fx.wrap
def fused_conv_sigmoid_mul_hardtanh(conv_input, weight, bias, mul_input):
    """Wrapper function to launch the fused kernel"""
    
    # Get tensor shapes
    batch_size, _, height, width = conv_input.shape
    out_channels = weight.shape[0]  # weight shape: [out_channels, in_channels, 1, 1]
    in_channels = weight.shape[1]
    
    # Create output tensor (same shape as mul_input)
    output = torch.empty_like(mul_input)
    
    # Calculate grid dimensions:
    # batch_size: number of batches
    # out_channels: number of output channels  
    # num_threads: one thread per spatial position (height * width)
    num_threads = height * width
    
    grid = (batch_size, out_channels, num_threads)
    
    conv_stride_w, conv_stride_h = 1, 1  # Conv2D stride from original computation: (1, 1)
    
    # Launch kernel - BLOCK_SIZE is now 1 (one element per thread)
    BLOCK_SIZE = 1
    
    fused_conv_sigmoid_mul_hardtanh_kernel[grid](
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        mul_input_ptr=mul_input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        conv_stride_w=conv_stride_w,
        conv_stride_h=conv_stride_h,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_sigmoid_mul_hardtanh