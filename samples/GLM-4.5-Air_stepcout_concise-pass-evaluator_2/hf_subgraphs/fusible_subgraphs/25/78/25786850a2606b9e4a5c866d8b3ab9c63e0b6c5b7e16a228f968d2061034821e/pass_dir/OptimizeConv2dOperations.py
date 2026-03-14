import torch
import triton
import triton.language as tl

@triton.jit
def simple_conv_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride, padding, output_size,
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles one output location
    block_start = pid * block_size
    for i in range(block_size):
        program_idx = block_start + i
        if program_idx < output_size:
            # Calculate output position
            channel_idx = program_idx % out_channels
            h_idx = (program_idx // out_channels) % output_size
            w_idx = (program_idx // out_channels) // output_size
            
            # Initialize with bias
            result = tl.load(bias_ptr + channel_idx)
            
            # Simple convolution computation
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = h_idx * stride[0] + kh - padding[0]
                    iw = w_idx * stride[1] + kw - padding[1]
                    
                    if 0 <= ih < in_height and 0 <= iw < in_width:
                        for ic in range(in_channels):
                            input_idx = (ic * in_height + ih) * in_width + iw
                            weight_idx = (channel_idx * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw
                            
                            input_val = tl.load(input_ptr + input_idx)
                            weight_val = tl.load(weight_ptr + weight_idx)
                            result += input_val * weight_val
            
            tl.store(output_ptr + program_idx, result)

@torch.fx.wrap
def optimized_conv2d(input, weight, bias, stride, padding, dilation, groups):
    """
    Optimized conv2d with flattening and transpose in one pass
    """
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) // stride[1] + 1
    
    # For our specific case: stride=(16,16), padding=(0,0), kernel_size=16
    # Output: [1, 3, 224, 224] -> [1, 768, 14, 14]
    output_spatial = out_height * out_width
    total_elements = batch_size * out_channels * output_spatial
    
    # Create output tensor flattened: [1, 768, 14, 14] -> [1, 768*14*14]
    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    block_size = 256
    grid_size = (triton.cdiv(total_elements, block_size),)
    
    simple_conv_kernel[grid_size](
        input, weight, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding, total_elements,
        block_size
    )
    
    # Reshape back: [1, 768*14*14] -> [1, 768, 14, 14] -> flatten(2) -> transpose
    output = output.view(batch_size, out_channels, out_height, out_width)
    output = output.flatten(2)  # [1, 768, 196]
    output = output.transpose(1, 2)  # [1, 196, 768]
    
    return output

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match: conv2d + flatten + transpose operations
    """
    tmp_5 = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1)
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1)

# Replacement function
def replacement_func():
    return optimized_conv2d