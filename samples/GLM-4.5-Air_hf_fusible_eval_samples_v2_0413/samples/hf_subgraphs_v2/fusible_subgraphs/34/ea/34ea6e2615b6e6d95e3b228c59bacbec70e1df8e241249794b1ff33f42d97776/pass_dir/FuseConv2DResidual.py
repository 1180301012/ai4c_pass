import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_residual_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    n_elements = batch_size * out_channels * height * width
    
    if pid * BLOCK_SIZE >= n_elements:
        return
        
    # Calculate linear index and map to output coordinates
    linear_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < n_elements
    
    # Convert linear index to output coordinates: [batch, channel, h, w]
    output_coords = linear_idx // (height * width)
    channel_idx = (linear_idx % (height * width)) // width
    h_idx = (linear_idx % (height * width)) % width
    
    # Calculate input coordinates with padding and stride
    input_h = h_idx * stride_h - pad_h
    input_w = h_idx * stride_w - pad_w
    
    # Initialize output with bias and residual connection (input)
    linear_input_idx = output_coords * (in_channels * height * width) + channel_idx * (height * width) + h_idx * width + input_w
    
    # Load input (for residual connection)
    input_val = tl.load(input_ptr + linear_idx, mask=mask, other=0.0)
    
    # Apply depthwise convolution (groups == out_channels means depthwise)
    if groups == out_channels:
        # Depthwise convolution - each input channel convolved with its own kernel
        conv_result = input_val  # Start with input value
        
        # Apply 3x3 convolution manually for depthwise case
        for kh in range(kernel_size[0]):
            for kw in range(kernel_size[1]):
                # Calculate weight position for this channel and kernel position
                weight_offset = channel_idx * (kernel_size[0] * kernel_size[1]) + kh * kernel_size[1] + kw
                
                # Calculate input positions for this kernel offset
                input_h_offset = input_h + kh
                input_w_offset = input_w + kw
                
                # Check bounds and load weight and input
                if (0 <= input_h_offset < height and 0 <= input_w_offset < width):
                    input_offset = output_coords * (in_channels * height * width) + channel_idx * (height * width) + input_h_offset * width + input_w_offset
                    weight_val = tl.load(weight_ptr + weight_offset, mask=mask)
                    input_val_offset = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                    conv_result += weight_val * input_val_offset
                    
        # Add bias
        if bias_ptr is not None:
            bias_offset = channel_idx
            bias_val = tl.load(bias_ptr + bias_offset, mask=mask)
            conv_result += bias_val
            
        # Add residual connection (original input)
        output_result = conv_result + input_val
    else:
        # Regular convolution (simplified case)
        output_result = input_val  # Simplified for this example
    
    # Store result
    tl.store(output_ptr + linear_idx, output_result, mask=mask)

@torch.fx.wrap
def fused_conv2d_residual(input_tensor, weight_tensor, bias_tensor, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=768):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, k_in_channels, kernel_size_h, kernel_size_w = weight_tensor.shape
    
    if groups != out_channels:
        raise ValueError("This optimized version only supports depthwise convolution (groups == out_channels)")
    
    if k_in_channels != in_channels or kernel_size_h != 3 or kernel_size_w != 3 or stride != (1, 1) or padding != (1, 1):
        raise ValueError("This optimized version only supports specific 3x3 depthwise convolution parameters")
    
    # Create output tensor with same shape as input (for depthwise conv with stride 1, padding 1)
    output = torch.empty_like(input_tensor)
    
    # Kernel launch parameters
    n_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_conv2d_residual_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        batch_size, in_channels, out_channels, height, width,
        (3, 3),  # kernel_size
        stride[0], stride[1],  # stride_h, stride_w
        padding[0], padding[1],  # pad_h, pad_w
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Pattern matching function - matches Conv2D + residual
def pattern(input_tensor, weight_tensor, bias_tensor):
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (1, 1), (1, 1), 768)
    tmp_5 = conv2d + input_tensor
    return conv2d, tmp_5

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_forward(input_tensor, weight_tensor, bias_tensor):
        # Fuse conv2d and residual into single operation
        tmp_5 = fused_conv2d_residual(input_tensor, weight_tensor, bias_tensor, (1, 1), (1, 1), (1, 1), 768)
        # For the conv2d tensor that was used later, we need to extract it
        # Since we fused the operations, we can compute it separately or modify the pattern
        conv2d = tmp_5 - input_tensor  # Extract conv2d result from residual
        return conv2d, tmp_5
    
    return optimized_forward