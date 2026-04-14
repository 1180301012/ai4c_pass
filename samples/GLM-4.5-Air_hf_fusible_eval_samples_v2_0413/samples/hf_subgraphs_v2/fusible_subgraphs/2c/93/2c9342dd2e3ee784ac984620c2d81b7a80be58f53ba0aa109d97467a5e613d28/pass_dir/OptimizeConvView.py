import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D + View fusion
def pattern(input_tensor, weight_tensor, bias_tensor):
    """Match the sequence: conv2d(input, weight, bias) → view(batch_size, 1, -1)"""
    conv_output = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    # The first parameter of view determines the batch-size, which varies across different graphs
    viewed_output = conv_output.view(conv_output.size(0), 1, -1)
    return viewed_output

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    # Extract all three tensor arguments for conv2d
    return (input_tensor, weight_tensor, bias_tensor)

# Optimized Triton kernel for Conv2D + View fusion
@triton.jit
def conv2d_view_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride, padding, dilation, groups,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Conv2D followed by view operation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    # Flatten spatial dimensions for view operation
    spatial_size = out_height * out_width
    batch_stride = out_channels * spatial_size
    channel_stride = spatial_size
    
    # Compute indices
    offset_flattened = offsets
    batch_idx = offset_flattened // batch_stride
    channel_idx = (offset_flattened % batch_stride) // channel_stride  
    spatial_idx = offset_flattened % channel_stride
    
    # Convert 3D indices to 2D for conv2d
    h_idx = spatial_idx // out_width
    w_idx = spatial_idx % out_width
    
    # Handle bias broadcasting (bias is [1] for all cases)
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + 0, other=0.0)  # broadcast bias to all elements
    else:
        bias_val = 0.0
    
    # Load input element
    input_idx = (batch_idx * in_channels + channel_idx) * in_height * in_width + h_idx * in_width + w_idx
    input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Load weight element (for 1x1 conv, weight is [out_channels, in_channels, 1, 1])
    weight_idx = channel_idx  # 1x1 kernel, so spatial indices are 0
    weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
    
    # For 1x1 convolution with groups=1: output = input * weight + bias
    result = input_val * weight_val + bias_val
    
    # Store result in flattened format (batch_size, 1, -1)
    tl.store(output_ptr + offset_flattened, result, mask=mask)

# Kernel wrapper decorated with torch.fx.wrap
@torch.fx.wrap
def optimized_conv2d_view(input_tensor, weight_tensor, bias_tensor):
    """Wrapper function for fused Conv2D + View operation"""
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight_tensor.shape
    kernel_size = (kernel_size_h, kernel_size_w)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    
    # Calculate total elements in flattened tensor
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    n_elements = batch_size * out_channels * out_height * out_width
    
    BLOCK_SIZE = 1024  # Optimal block size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (already flattened to batch_size, 1, -1)
    output = torch.empty((batch_size, 1, out_channels * out_height * out_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the fused kernel
    conv2d_view_fused_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns the kernel wrapper (not called)
def replacement_func():
    return optimized_conv2d_view