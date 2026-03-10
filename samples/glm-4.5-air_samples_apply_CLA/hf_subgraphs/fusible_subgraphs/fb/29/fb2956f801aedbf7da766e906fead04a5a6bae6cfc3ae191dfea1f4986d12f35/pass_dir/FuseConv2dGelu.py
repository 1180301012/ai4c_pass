import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    tmp_2 = torch.conv2d(x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    tmp_3 = torch.nn.functional.gelu(tmp_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_3, tmp_4

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def conv2d_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, 
    out_ptr,
    batch_size, in_channels, height, width,
    out_channels, kernel_size, padding,
    stride, groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one output channel
    pid = tl.program_id(0)
    total_channels = out_channels if groups == 1 else out_channels // groups
    channel_idx = pid % total_channels
    
    # Only proceed if this channel is valid
    if channel_idx >= out_channels:
        return
        
    # Input strides
    x_stride_n = in_channels * height * width
    x_stride_c = height * width
    x_stride_h = width
    x_stride_w = 1
    
    # Weight strides  
    w_out_c = groups * out_channels * kernel_size * kernel_size
    w_in_c = kernel_size * kernel_size
    w_h = kernel_size
    w_w = 1
    
    # Output strides
    out_stride_n = out_channels * height * width
    out_stride_c = height * width
    out_stride_h = width
    out_stride_w = 1
    
    # Calculate output position
    out_offset = channel_idx * out_stride_c
    out_h_base = (out_offset // out_stride_w) // out_stride_h
    out_c_base = out_offset // out_stride_c
    
    # Process spatial dimensions in tiles
    for h_idx in tl.range(0, height, BLOCK_SIZE_M):
        for w_idx in tl.range(0, width, BLOCK_SIZE_N):
            h_start = h_idx
            h_end = min(h_start + BLOCK_SIZE_M, height)
            w_start = w_idx
            w_end = min(w_start + BLOCK_SIZE_N, width)
            
            # Initialize accumulator for this tile
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            # Process input channels
            for in_c in range(in_channels):
                # Get weight for this input channel and output channel
                if groups == 1:
                    weight_offset = (channel_idx * in_channels + in_c) * kernel_size * kernel_size
                else:
                    group_id = channel_idx // (out_channels // groups)
                    local_channel = channel_idx % (out_channels // groups)
                    weight_offset = (group_id * in_channels + in_c + local_channel * in_channels) * kernel_size * kernel_size
                
                w = tl.load(weight_ptr + weight_offset)
                
                # Load weight kernel
                weight_values = []
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        weight_offset_w = weight_offset + kh * w_h + kw * w_w
                        weight_values.append(tl.load(weight_ptr + weight_offset_w))
                weight_tensor = tl.tensor(weight_values, dtype=tl.float32).reshape(kernel_size, kernel_size)
                
                # Convolve weight with input patch
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Calculate input position with padding
                        in_h = h_start + kh - padding
                        in_w = w_start + kw - padding
                        
                        if 0 <= in_h < height and 0 <= in_w < width:
                            x_offset = in_c * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                            x_val = tl.load(x_ptr + x_offset)
                            acc += x_val * weight_tensor[kh, kw]
            
            # Add bias if present
            if bias_ptr is not None:
                if groups == 1:
                    bias_offset = channel_idx
                else:
                    group_id = channel_idx // (out_channels // groups)
                    local_channel = channel_idx % (out_channels // groups)
                    bias_offset = group_id * (out_channels // groups) + local_channel
                bias_val = tl.load(bias_ptr + bias_offset)
                acc += bias_val
            
            # Apply GELU
            acc = acc * 0.5 * (1.0 + tl.tanh(acc * 0.7978845608 * (1.0 + 0.044715 * acc * acc)))
            
            # Store results
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    if groups == 1:
                        out_channel_idx = channel_idx
                    else:
                        group_id = channel_idx // (out_channels // groups)
                        local_channel = channel_idx % (out_channels // groups)
                        out_channel_idx = group_id * (out_channels // groups) + local_channel
                    
                    if out_channel_idx < out_channels:
                        out_offset = out_channel_idx * out_stride_c + h * out_stride_h + w * out_stride_w
                        tl.store(out_ptr + out_offset, acc[h - h_start, w - w_start])

@torch.fx.wrap
def triton_conv2d_gelu(x, weight, bias):
    batch_size, in_channels, height, width = x.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = weight.shape
    
    # Handle grouped convolution
    groups = weight.shape[0] if len(weight.shape) == 5 else 1
    if groups == 1:
        out_channels = weight.shape[0]
        in_channels = weight.shape[1]
    else:
        out_channels = weight.shape[0]
        kernel_in_channels = weight.shape[2]
    
    padding = 1
    stride = 1
    
    # Output size calculation
    out_height = (height + 2 * padding - kernel_h) // stride + 1
    out_width = (width + 2 * padding - kernel_w) // stride + 1
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Determine block sizes based on tensor dimensions
    BLOCK_SIZE_M = 16  # Height tile size
    BLOCK_SIZE_N = 16  # Width tile size
    BLOCK_SIZE_K = min(32, in_channels)  # Channel tile size
    
    # Number of programs needed
    total_channels = out_channels if groups == 1 else out_channels * batch_size
    num_programs = (total_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    conv2d_gelu_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        kernel_size=kernel_h,
        padding=padding,
        stride=stride,
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return triton_conv2d_gelu