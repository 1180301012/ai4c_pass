import torch
import triton
import triton.language as tl

def pattern(in_6, in_1, in_0):
    # Match the conv3d -> flatten -> transpose sequence
    conv3d = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8

def replacement_args(in_6, in_1, in_0):
    return (in_6, in_1, in_0)

@triton.jit
def simple_conv3d_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_d,
    input_h,
    input_w,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    output_d,
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position and channel
    spatial_size = output_d * output_h * output_w
    channel_size = out_channels
    
    pid = tl.program_id(0)
    
    # Only process spatial positions within valid range
    if pid >= spatial_size:
        return
    
    # Determine which channel this program handles
    c_out = pid % channel_size
    spatial_offset = pid // out_channels
    
    # Decode spatial coordinates from the spatial offset
    if spatial_offset >= output_d * output_h * output_w:
        return
        
    d = spatial_offset // (output_h * output_w)
    rem = spatial_offset % (output_h * output_w)
    h = rem // output_w
    w = rem % output_w
    
    # Initialize accumulator for this spatial position and channel
    acc = 0.0
    
    # Load bias for this channel (with proper mask for c_out)
    c_out_mask = c_out < out_channels
    bias = tl.load(bias_ptr + c_out, mask=c_out_mask, other=0.0)
    
    # Compute convolution sum
    for c_in in range(in_channels):
        for dd in range(kernel_d):
            for hh in range(kernel_h):
                for ww in range(kernel_w):
                    # Input coordinates
                    in_d = d * stride_d + dd
                    in_h = h * stride_h + hh
                    in_w = w * stride_w + ww
                    
                    # Only proceed if input coordinates are valid
                    if (0 <= in_d) and (in_d < input_d) and (0 <= in_h) and (in_h < input_h) and (0 <= in_w) and (in_w < input_w):
                        # Calculate input position
                        in_idx = in_d * (input_h * input_w) + in_h * input_w + in_w
                        input_pos = in_idx * in_channels + c_in
                        
                        # Calculate weight position (assuming contiguous layout)
                        weight_idx = (c_out * in_channels + c_in) * kernel_d * kernel_h * kernel_w + \
                                    dd * kernel_h * kernel_w + hh * kernel_w + ww
                        
                        # Load values with proper masks
                        input_val = tl.load(input_ptr + input_pos, other=0.0)
                        c_in_mask = c_in < in_channels
                        weight_val = tl.load(weight_ptr + weight_idx, mask=c_in_mask, other=0.0)
                        
                        # Add to accumulator
                        acc += input_val * weight_val
    
    # Add bias
    acc += bias
    
    # Calculate output position (spatial flattened, then channel)
    output_pos = spatial_offset * out_channels + c_out
    
    # Store result
    tl.store(output_ptr + output_pos, acc)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(input, weight, bias):
    # Input shapes: input [B,C,D,H,W], weight [Co,Ci,Dk,Hk,Wk], bias [Co]
    B, C, D, H, W = input.shape
    Co, Ci, Dk, Hk, Wk = weight.shape
    
    # Compute output dimensions
    D_out = (D - Dk) // 2 + 1  # stride = 2
    H_out = (H - Hk) // 16 + 1  # stride = 16  
    W_out = (W - Wk) // 16 + 1  # stride = 16
    
    # Create output tensor [B, D_out*H_out*W_out, Co]
    output = torch.empty([B, D_out * H_out * W_out, Co], dtype=input.dtype, device=input.device)
    
    # Grid configuration - use 1D grid for spatial positions
    spatial_size = D_out * H_out * W_out
    BLOCK_SIZE = 1024
    grid = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
    
    # Launch kernel
    simple_conv3d_kernel[grid](
        input,
        weight,
        bias,
        output,
        B, C, Co,
        D, H, W,
        Dk, Hk, Wk,
        2, 16, 16,  # strides
        D_out, H_out, W_out,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv3d_flatten_transpose