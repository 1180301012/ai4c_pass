import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match Conv2D + no-op multiplication + reshape pattern"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the optimized kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_reshape_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    hin: tl.constexpr,
    win: tl.constexpr,
    channels_in: tl.constexpr,
    channels_out: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that fuses Conv2D and reshape operations"""
    pid = tl.program_id(0)
    
    # Calculate which output element this thread handles
    total_elements = batch_size * out_h * out_w
    if pid >= total_elements:
        return
    
    # Decode batch position and spatial position
    batch_idx = pid // (out_h * out_w)
    spatial_pid = pid % (out_h * out_w) 
    h_idx = spatial_pid // out_w
    w_idx = spatial_pid % out_w
    
    # Each thread computes one output channel - but we have 17 channels
    channel_idx = pid % channels_out
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Conv2D calculation for this (batch, h, w, channel) combination
    output_val = bias_val
    
    # Weight shape: [channels_out, channels_in, kernel_h, kernel_w]
    weight_offset = channel_idx * channels_in * kernel_h * kernel_w
    
    for ci in range(channels_in):
        input_channel_offset = batch_idx * channels_in * hin * win + ci * hin * win
        weight_ci_offset = weight_offset + ci * kernel_h * kernel_w
        
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input coordinates with bounds checking
                ih = h_idx + kh
                iw = w_idx + kw
                if ih < hin and iw < win:
                    # Load input value
                    input_elem_offset = input_channel_offset + ih * win + iw
                    input_val = tl.load(input_ptr + input_elem_offset)
                    
                    # Load weight value
                    weight_elem_offset = weight_ci_offset + kh * kernel_w + kw
                    weight_val = tl.load(weight_ptr + weight_elem_offset)
                    
                    output_val += input_val * weight_val
    
    # Store result: each thread stores one value in the flattened output
    tl.store(output_ptr + pid, output_val)

@torch.fx.wrap
def optimized_conv2d_reshape(bias, weight, input_tensor):
    """Wrapper function for the optimized Conv2D + reshape kernel"""
    batch_size, channels_in, hin, win = input_tensor.shape
    channels_out = bias.shape[0]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    # Output dimensions after Conv2D with stride 1, padding 0, dilation 1
    out_h = hin
    out_w = win
    
    # Total number of output values to compute
    total_elements = batch_size * out_h * out_w * channels_out
    
    # Create output tensor  
    output = torch.empty(total_elements, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    # Each computes one output element using one program
    num_programs = total_elements
    conv2d_reshape_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        hin=hin,
        win=win,
        channels_in=channels_in,
        channels_out=channels_out,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        out_h=out_h,
        out_w=out_w,
        BLOCK_SIZE=1,  # Each program handles one element
    )
    
    # Reshape to the final target pattern
    output = output.reshape(batch_size, channels_out, out_h, out_w)  # [batch_size, 17, 64, 64]
    output = output.reshape(-1, 17, 4096)  # Final reshape
    
    return output

def replacement_func():
    """Return the optimized kernel function"""
    return optimized_conv2d_reshape