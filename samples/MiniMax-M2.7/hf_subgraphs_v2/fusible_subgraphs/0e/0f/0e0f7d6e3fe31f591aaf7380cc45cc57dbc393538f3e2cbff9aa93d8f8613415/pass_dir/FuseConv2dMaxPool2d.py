import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_maxpool2d_kernel(
    # Conv2d inputs
    input_ptr, weight_ptr, 
    # Conv2d params (stride, padding, dilation, groups)
    conv_stride_h, conv_stride_w,
    conv_padding_h, conv_padding_w, 
    conv_dilation_h, conv_dilation_w,
    conv_groups,
    # Input shape
    batch, in_channels, in_h, in_w,
    # Weight shape
    out_channels, kH, kW,
    # Output shape
    out_h, out_w,
    # Output pointer
    output_ptr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Data type
    DTYPE: tl.constexpr
):
    """
    Fused Conv2d + MaxPool2d kernel.
    Conv2d with stride S, padding P, dilation D, groups G
    Then MaxPool2d with kernel_size=3, stride=2, padding=1
    
    For each output pixel in the pooled grid:
    - Compute the corresponding conv output region
    - Find the max value in that 3x3 region
    """
    # Get current program id
    pid = tl.program_id(0)
    
    # Calculate which output position this program handles
    # Output is (batch, out_channels, out_h, out_w)
    num_channels = out_channels
    num_h = out_h
    num_w = out_w
    
    # Calculate position
    channel_id = pid // (num_h * num_w)
    remaining = pid % (num_h * num_w)
    h_id = remaining // num_w
    w_id = remaining % num_w
    
    # Bounds check
    if channel_id >= num_channels or h_id >= num_h or w_id >= num_w:
        return
    
    # Calculate the starting position in input for this output
    # For conv: output = (input + 2*padding - dilation*(kernel-1) - 1) / stride + 1
    # Inverted: input_start = h_id * conv_stride - conv_padding for h dimension after pool
    # After pool, output h_id corresponds to conv output h = h_id * 2 - 1 (with pad=1)
    # So conv output h = h_id * 2 - 1 + 1 = h_id * 2
    # Therefore conv input h = h_id * 2 * stride_h - padding_h
    
    conv_h_start = h_id * 2 * conv_stride_h - conv_padding_h
    conv_w_start = w_id * 2 * conv_stride_w - conv_padding_w
    
    # Each output pixel needs to find max over a 3x3 region of conv output
    # The conv output at position (conv_h, conv_w) is the max over:
    # sum over kH x kW of input[conv_h + kh] * weight[kh, kw]
    
    # We'll compute the max pool by:
    # 1. For each position in the 3x3 pool window on conv output
    # 2. Compute the conv value
    # 3. Take the maximum
    
    max_val = float("-inf") if DTYPE != 2 else float("-inf")  # inf for bf16
    
    # Iterate over 3x3 pool window
    for pool_h_offset in range(3):
        for pool_w_offset in range(3):
            conv_h = conv_h_start + pool_h_offset * conv_stride_h
            conv_w = conv_w_start + pool_w_offset * conv_stride_w
            
            # Check if this conv position is valid (padding handling)
            # Conv with padding - positions outside input are treated as 0
            valid = True
            if conv_h < 0 or conv_w < 0 or conv_h >= in_h or conv_w >= in_w:
                valid = False
            
            # Compute conv value for this position
            if valid:
                # Convolution: sum over in_channels x kH x kW
                conv_val = tl.zeros((1,), tl.float32)
                
                # Group convolution - each group handles in_channels/groups channels
                channels_per_group = in_channels // conv_groups
                out_channels_per_group = out_channels // conv_groups
                
                # Determine which group this output channel belongs to
                group_id = channel_id // out_channels_per_group
                channel_in_group = channel_id % out_channels_per_group
                
                # Accumulate convolution
                for k in range(0, kH * kW, BLOCK_K):
                    kH_offset = k // kW
                    kW_offset = k % kW
                    
                    # Load weight for this kernel position
                    weight_offset = (
                        channel_in_group * (in_channels // conv_groups) * kH * kW +
                        0 * kH * kW +
                        (kH_offset * kW + kW_offset)
                    )
                    w = tl.load(weight_ptr + weight_offset)
                    
                    # Load input for all channels in group
                    for c in range(channels_per_group):
                        inp_channel = group_id * channels_per_group + c
                        
                        inp_h = conv_h + kH_offset * conv_dilation_h
                        inp_w = conv_w + kW_offset * conv_dilation_w
                        
                        inp_offset = (
                            0 * in_channels * in_h * in_w +
                            inp_channel * in_h * in_w +
                            inp_h * in_w +
                            inp_w
                        )
                        x = tl.load(input_ptr + inp_offset)
                        
                        conv_val += x * w
                
                # Apply relu if needed (conv output can go negative)
                # Actually, the conv output just needs max pooling
                # The pool window is 3x3 so we take max over it
                
            else:
                conv_val = float("-inf")  # Padding is 0 for conv, but for max pool we use -inf
            
            max_val = tl.max(max_val, conv_val if valid else float("-inf"))
    
    # Store result
    output_offset = (
        0 * num_channels * num_h * num_w +
        channel_id * num_h * num_w +
        h_id * num_w +
        w_id
    )
    tl.store(output_ptr + output_offset, max_val)


def fused_conv2d_maxpool2d(
    input_tensor, weight,
    stride, padding, dilation, groups
):
    """
    Fused Conv2d + MaxPool2d operation.
    MaxPool2d parameters: kernel_size=3, stride=2, padding=1
    """
    # Get shapes
    batch, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate conv output size
    # Formula: output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    conv_out_h = (in_h + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    conv_out_w = (in_w + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    
    # Calculate max pool output size
    # Formula: output_size = floor((input_size + 2*padding - kernel_size) / stride + 1)
    pool_kernel = 3
    pool_stride = 2
    pool_padding = 1
    out_h = (conv_out_h + 2 * pool_padding - pool_kernel) // pool_stride + 1
    out_w = (conv_out_w + 2 * pool_padding - pool_kernel) // pool_stride + 1
    
    # Allocate output
    output = torch.empty(
        (batch, out_channels, out_h, out_w),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Grid size: batch * out_channels * out_h * out_w
    grid_size = batch * out_channels * out_h * out_w
    
    if grid_size == 0:
        return output
    
    # Block sizes
    BLOCK_M = 1
    BLOCK_N = 1
    BLOCK_K = 16
    
    # Launch kernel
    fused_conv2d_maxpool2d_kernel[(grid_size,)](
        input_tensor, weight,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        batch, in_channels, in_h, in_w,
        out_channels, kH, kW,
        out_h, out_w,
        output,
        BLOCK_M, BLOCK_N, BLOCK_K,
        1 if input_tensor.dtype == torch.float32 else (2 if input_tensor.dtype == torch.bfloat16 else 0)
    )
    
    return output


def pattern(in_0, in_1):
    """
    Match pattern: Conv2d followed by MaxPool2d with kernel=3, stride=2, pad=1
    """
    # Extract stride, padding, dilation, groups from the conv2d call
    # The pattern: torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    tmp = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    result = torch.nn.functional.max_pool2d(tmp, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv2d_maxpool2d