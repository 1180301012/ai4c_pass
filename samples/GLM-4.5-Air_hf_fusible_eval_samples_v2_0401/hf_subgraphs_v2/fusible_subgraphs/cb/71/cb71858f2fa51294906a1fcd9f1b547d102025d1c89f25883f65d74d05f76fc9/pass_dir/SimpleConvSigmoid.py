import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def simple_conv_sigmoid_kernel(
    weight_ptr, bias_ptr, x_ptr, out_ptr,
    batch, out_c, in_c, h_in, w_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elems = batch * 2 * 8 * 8  # Final output shape: [1, 2, 8, 8]
    mask = pid < n_elems
    
    # Map final position to conv2d output position
    b = pid // (2 * 8 * 8)
    rem = pid % (2 * 8 * 8)
    final_c = rem // (8 * 8)  # 0 or 1
    rem2 = rem % (8 * 8)
    final_h = rem2 // 8  # 0-7  
    final_w = rem2 % 8   # 0-7
    
    # Map final position to original conv2d output channel
    # Original conv2d: input [1, 2, 1, 8] x weights [128, 2, 1, 8] -> output [1, 128, 1, 8]
    # Then view(1, 2, 8, 8) maps 128 channels to 2 groups * 8*8 spatial locations = 128 total
    # Each final channel (0-1) gets 64 spatial locations (8x8)
    channels_per_group = 64  # 128 total / 2 final channels
    base_channel = final_c * channels_per_group
    local_spatial_pos = final_h * 8 + final_w
    conv_c = base_channel + local_spatial_pos
    
    # Conv2D computation for this single output position
    result = 0.0
    for ic in range(in_c):  # 2 input channels
        for kh in range(h_in):  # 1 height 
            for kw in range(w_in):  # 8 width
                # Load weight: [out_channels, in_channels, kh, kw] = [128, 2, 1, 8]
                weight_offset = conv_c * in_c * h_in * w_in + ic * h_in * w_in + kh * w_in + kw
                weight = tl.load(weight_ptr + weight_offset, 
                               mask=(conv_c < out_c) & (ic < in_c), 
                               other=0.0)
                
                # Load input: [batch, in_channels, height, width] = [1, 2, 1, 8]
                # For output position (b, conv_c, 0, final_w), we need input from (b, ic, 0, final_w) 
                # because stride=(1,1), padding=(0,0) means spatial alignment
                input_offset = b * in_c * h_in * w_in + ic * h_in * w_in + kh * w_in + final_w
                input_val = tl.load(x_ptr + input_offset, 
                                  mask=(b < batch) & (ic < in_c), 
                                  other=0.0)
                
                result += weight * input_val
    
    # Add bias
    bias = tl.load(bias_ptr + conv_c, mask=(conv_c < out_c), other=0.0)
    result += bias
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-result))
    
    # Store result
    tl.store(out_ptr + pid, sigmoid_val, mask=mask)

@torch.fx.wrap
def simple_conv_sigmoid(in_0, in_1, in_2):
    batch, in_c, h_in, w_in = in_2.shape
    out_c = in_1.shape[0]
    
    total_elems = batch * 2 * 8 * 8
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elems, BLOCK_SIZE),)
    
    output = torch.empty((batch, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)
    
    in_1 = in_1.to(in_2.device)
    in_0 = in_0.to(in_2.device)
    
    simple_conv_sigmoid_kernel[grid_size](
        in_1, in_0, in_2, output,
        batch, out_c, in_c, h_in, w_in,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_conv_sigmoid