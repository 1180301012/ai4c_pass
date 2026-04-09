import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the redundant pattern: conv2d -> stack -> sum -> concat"""
    # Match the exact computation from the model
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_3], 1)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for replacement"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_conv2d_cat_kernel(
    conv_input_ptr,      # Input to conv2d
    weight_ptr,          # Conv weights
    bias_ptr,            # Conv bias
    concat_tensor_ptr,   # Second input to concatenation
    output_ptr,
    batch_size,          # Number of batches
    in_channels,         # Input channels
    out_channels,        # Output channels  
    height,              # Height of feature map
    width,               # Width of feature map,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Get program ID and compute coordinates
    pid = tl.program_id(0)
    
    # Calculate batch, spatial coordinates
    total_pixels = batch_size * height * width
    if pid >= total_pixels:
        return
        
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Process all output channels for this pixel
    for oc in range(out_channels):
        # Calculate conv result for this output channel
        # Start with bias
        conv_result = tl.load(bias_ptr + oc)
        
        # Compute weighted sum of input channels (1x1 convolution)
        for ic in range(in_channels):
            # Input tensor offset: [batch, channel, h, w]
            input_offset = (batch_idx * in_channels + ic) * height * width + h_idx * width + w_idx
            input_val = tl.load(conv_input_ptr + input_offset)
            
            # Weight tensor offset: [out_channel, in_channel]  
            weight_offset = oc * in_channels + ic
            weight_val = tl.load(weight_ptr + weight_offset)
            
            conv_result += input_val * weight_val
        
        # Store conv result in first half of output
        output_offset = batch_idx * out_channels * 2 * height * width + oc * height * width + h_idx * width + w_idx
        tl.store(output_ptr + output_offset, conv_result)
    
    # Copy concatenation tensor to second half of output
    for oc_concat in range(out_channels):
        concat_offset = batch_idx * out_channels * height * width + oc_concat * height * width + h_idx * width + w_idx
        output_offset = batch_idx * out_channels * 2 * height * width + (out_channels + oc_concat) * height * width + h_idx * width + w_idx
        concat_val = tl.load(concat_tensor_ptr + concat_offset)
        tl.store(output_ptr + output_offset, concat_val)

@torch.fx.wrap  
def optimized_conv2d_cat(conv_bias, conv_weight, conv_input, concat_tensor):
    # Handle tensor shapes correctly - bias is 1D, others are 4D
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Output has 2 * out_channels from concatenation  
    output_shape = (batch_size, out_channels * 2, height, width)
    output = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Flatten input tensors to 1D for easier pointer arithmetic
    conv_input_flat = conv_input.reshape(-1)
    conv_weight_flat = conv_weight.reshape(-1)
    concat_tensor_flat = concat_tensor.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Grid configuration - one program per spatial pixel in each batch
    total_pixels = batch_size * height * width
    grid = (total_pixels,)
    
    # Launch optimized kernel with simpler block configuration
    optimized_conv2d_cat_kernel[grid](
        conv_input_ptr=conv_input_flat,
        weight_ptr=conv_weight_flat,
        bias_ptr=conv_bias,
        concat_tensor_ptr=concat_tensor_flat,
        output_ptr=output_flat,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=1,
    )
    
    return output

def replacement_func():
    """Return the optimized function reference"""
    return optimized_conv2d_cat