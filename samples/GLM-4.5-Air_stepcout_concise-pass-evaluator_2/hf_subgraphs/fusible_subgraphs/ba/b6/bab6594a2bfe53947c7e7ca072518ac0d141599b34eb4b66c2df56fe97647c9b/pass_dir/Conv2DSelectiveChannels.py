import torch
import triton
import triton.language as tl

@triton.jit
def triton_conv2d_elementwise_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w,
    BLOCK_SIZE: tl.constexpr
):
    """Element-wise convolution kernel for 1x1 kernels"""
    pid = tl.program_id(0)
    
    # Calculate output dimensions  
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_size - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_size - 1) - 1) // stride_w + 1
    
    # Calculate which output position and channel this program handles
    total_channels = out_channels * out_height * out_width
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < total_channels
    
    if not mask.any():
        return
    
    # Convert linear position to multi-dimensional indices
    channel_idx = pos // (out_height * out_width)
    height_idx = (pos % (out_height * out_width)) // out_width
    width_idx = pos % out_width
    
    # Handle only valid positions
    valid_mask = (channel_idx < out_channels) & (height_idx < out_height) & (width_idx < out_width)
    pos_valid = tl.where(valid_mask, pos, -1)
    
    if not (pos_valid >= 0).any():
        return
    
    # Calculate input position for valid positions
    input_h = height_idx * stride_h - padding_h
    input_w = width_idx * stride_w - padding_w
    
    # Create input batch and channel indices
    batch_idx = tl.zeros_like(pos, dtype=tl.int32)  # Assuming batch_size = 1
    input_channel_idx = channel_idx  # For 1x1 conv with groups=1
    
    # Calculate input linear offset
    input_offset = batch_idx * in_channels * in_height * in_width + \
                   input_channel_idx * in_height * in_width + \
                   input_h.flatten() * in_width + input_w.flatten()
    
    # Mask for valid input positions
    input_valid_mask = (input_h >= 0) & (input_h < in_height) & (input_w >= 0) & (input_w < in_width)
    input_offset = tl.where(input_valid_mask.flatten(), input_offset, -1)
    
    # Load input values
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset >= 0, other=0.0)
    
    # Load weight values (for 1x1 conv, weight is just channel scaling)
    weight_offset = channel_idx.flatten()
    weight_vals = tl.load(weight_ptr + weight_offset, mask=channel_idx.flatten() < out_channels, other=0.0)
    
    # Load bias values
    bias_vals = tl.load(bias_ptr + channel_idx.flatten(), mask=channel_idx.flatten() < out_channels, other=0.0)
    
    # Perform convolution (for 1x1 it's just weighted sum)
    output_vals = input_vals * weight_vals + bias_vals
    
    # Store results
    tl.store(out_ptr + pos_valid, output_vals, mask=pos_valid >= 0)

@torch.fx.wrap
def conv2d_selective_channels(input, weight, bias, stride, padding, dilation, groups, selected_channels):
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else torch.zeros(weight.size(0), device=weight.device, dtype=weight.dtype)
    
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    # Handle stride, padding, dilation parameters
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
    dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    kernel_size = kernel_size_h
    
    # Compute output dimensions
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_size - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_size - 1) - 1) // stride_w + 1
    
    # Create output tensors
    full_output = torch.zeros((batch_size, out_channels, out_height, out_width), 
                            dtype=input.dtype, device=input.device)
    selected_output = torch.zeros((batch_size, selected_channels, out_height, out_width), 
                                dtype=input.dtype, device=input.device)
    
    # Triton launch configuration
    BLOCK_SIZE = 1024
    
    # Calculate total elements for output
    total_elements_full = batch_size * out_channels * out_height * out_width
    num_programs_full = (total_elements_full + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    total_elements_selected = batch_size * selected_channels * out_height * out_width
    num_programs_selected = (total_elements_selected + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for full output
    triton_conv2d_elementwise_kernel[(num_programs_full,)](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, out_ptr=full_output,
        batch_size=batch_size, in_channels=in_channels, in_height=in_height, in_width=in_width,
        out_channels=out_channels, kernel_size=kernel_size, 
        stride_h=stride_h, stride_w=stride_w,
        dilation_h=dilation_h, dilation_w=dilation_w, 
        padding_h=padding_h, padding_w=padding_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch kernel for selected output (only compute what we need)
    if selected_channels < out_channels:
        triton_conv2d_elementwise_kernel[(num_programs_selected,)](
            input_ptr=input, weight_ptr=weight, bias_ptr=bias, out_ptr=selected_output,
            batch_size=batch_size, in_channels=in_channels, in_height=in_height, in_width=in_width,
            out_channels=selected_channels, kernel_size=kernel_size, 
            stride_h=stride_h, stride_w=stride_w,
            dilation_h=dilation_h, dilation_w=dilation_w, 
            padding_h=padding_h, padding_w=padding_w,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        selected_output = full_output
    
    return selected_output, full_output

def pattern(in_1, in_0):
    """Match conv2d followed by channel slice"""
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = tmp_1[slice(None, None, None), slice(None, tmp_1.size(1)), slice(None, None, None), slice(None, None, None)]
    return tmp_2, tmp_1

def replacement_args(in_1, in_0):
    """Extract arguments for the optimized kernel"""
    # Extract stride from the conv2d operation
    stride = (1, 1)  # Default for this pattern
    return (in_1, in_0, None, stride, (0, 0), (1, 1), 1, in_1.size(1))

def replacement_func():
    """Return the optimized function reference"""
    return conv2d_selective_channels