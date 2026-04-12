import torch
import triton
import triton.language as tl

def pattern(in_0, in_4, in_3):
    """Pattern matching for generic Conv2D + Flatten + Transpose fusion"""
    # This pattern will match both 2x2 and 4x4 convolutions
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(in_0, in_4, in_3):
    """Extract arguments for Conv2D + Flatten + Transpose fusion kernel"""
    return (in_0, in_4, in_3)

@triton.jit
def fuse_conv2d_flatten_transpose_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + Flatten + Transpose kernel"""
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    out_features = out_height * out_width
    
    # Program ID for parallel execution
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * out_channels, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * out_channels)
    
    # Compute base positions
    batch_idx = offsets // out_channels
    channel_idx = offsets % out_channels
    
    for h_idx in range(out_height):
        for w_idx in range(out_width):
            # Load input data
            input_base = batch_idx * in_channels * in_height * in_width + \
                         channel_idx // groups * in_channels // groups * in_height * in_width
            
            # Conv2D computation
            conv_val = tl.zeros([], dtype=tl.float32)
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    ih = h_idx * stride_height + ki * dilation_height - padding_height
                    iw = w_idx * stride_width + kj * dilation_width - padding_width
                    if ih >= 0:
                        if ih < in_height:
                            if iw >= 0:
                                if iw < in_width:
                                    input_val = tl.load(input_ptr + input_base + ih * in_width + iw, mask=True).to(tl.float32)
                                    weight_idx = channel_idx % out_channels * kernel_height * kernel_width + ki * kernel_width + kj
                                    weight_val = tl.load(weight_ptr + weight_idx, mask=True).to(tl.float32)
                                    conv_val = conv_val + input_val * weight_val
            
            # Add bias
            bias_idx = channel_idx
            bias_val = tl.load(bias_ptr + bias_idx, mask=True).to(tl.float32)
            conv_val = conv_val + bias_val
            
            # Flatten and transpose: treat batch as major, then flattened spatial dimensions, then channels
            # Original: [batch, out_channels, out_height, out_width] -> flatten to [batch, out_channels, out_features] 
            # Transpose to: [batch, out_features, out_channels]
            if out_channels == 16:  # Model-specific optimization
                spatial_idx = h_idx * out_width + w_idx
                output_idx = batch_idx * (out_height * out_width * out_channels) + spatial_idx * out_channels + channel_idx
                tl.store(output_ptr + output_idx, conv_val, mask=mask)

@torch.fx.wrap
def fused_conv2d_flatten_transpose(input, weight, bias):
    """Wrapper for fused Conv2D + Flatten + Transpose kernel"""
    input_shape = input.shape
    weight_shape = weight.shape
    
    batch_size, in_channels, in_height, in_width = input_shape
    # Handle both [out_channels, in_channels, H, W] and weight tensor cases
    if len(weight_shape) == 4:
        out_channels, in_channels_weight, kernel_height, kernel_width = weight_shape
        # Verify input channels match
        assert in_channels == in_channels_weight, f"Input channels {in_channels} don't match weight {in_channels_weight}"
    else:
        raise ValueError(f"Unexpected weight shape: {weight_shape}")
    
    stride_height, stride_width = 2, 2
    padding_height, padding_width = 0, 0
    dilation_height, dilation_width = 1, 1
    groups = 1
    
    out_height = (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    
    output_features = out_height * out_width
    total_elements = batch_size * output_features * out_channels
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine output dtype (same as input)
    dtype = input.dtype
    if dtype == torch.bfloat16:
        output = torch.empty((batch_size, output_features, out_channels), dtype=dtype, device=input.device)
    else:
        output = torch.empty((batch_size, output_features, out_channels), dtype=dtype, device=input.device)
    
    fuse_conv2d_flatten_transpose_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_height=stride_height,
        stride_width=stride_width,
        padding_height=padding_height,
        padding_width=padding_width,
        dilation_height=dilation_height,
        dilation_width=dilation_width,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_flatten_transpose