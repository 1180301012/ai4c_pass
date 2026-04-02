import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Conv-View-Sigmoid branch
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    
    # Sum-Division branch  
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    
    return (tmp_6, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def conv_sigmoid_kernel(
    weight_ptr, bias_ptr, x_ptr, out_ptr,
    batch, out_c, in_c, h_in, w_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elems = batch * 2 * 8 * 8
    mask = pid < n_elems
    
    # Map final position to conv2d output channel
    b = pid // (2 * 8 * 8)
    rem = pid % (2 * 8 * 8)
    final_c = rem // (8 * 8)  
    rem2 = rem % (8 * 8)
    final_h = rem2 // 8  
    final_w = rem2 % 8   
    
    # Channel mapping: 128 total channels → 2 groups * 8x8 spatial
    channels_per_group = 64
    base_channel = final_c * channels_per_group
    local_spatial_pos = final_h * 8 + final_w
    conv_c = base_channel + local_spatial_pos
    
    # Convolution
    result = 0.0
    for ic in range(in_c):
        for kh in range(h_in):
            for kw in range(w_in):
                weight_offset = conv_c * in_c * h_in * w_in + ic * h_in * w_in + kh * w_in + kw
                weight = tl.load(weight_ptr + weight_offset, mask=(conv_c < out_c) & (ic < in_c), other=0.0)
                
                input_offset = b * in_c * h_in * w_in + ic * h_in * w_in + kh * w_in + final_w
                input_val = tl.load(x_ptr + input_offset, mask=(b < batch) & (ic < in_c), other=0.0)
                
                result += weight * input_val
    
    # Add bias and sigmoid
    bias = tl.load(bias_ptr + conv_c, mask=(conv_c < out_c), other=0.0)
    result += bias
    sigmoid_val = 1.0 / (1.0 + tl.exp(-result))
    
    tl.store(out_ptr + pid, sigmoid_val, mask=mask)

@triton.jit
def sum_div_kernel(
    input_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Extract batch, channel, height
    bch_id = pid
    batch = bch_id // (channels * height)
    remainder = bch_id % (channels * height)
    channel = remainder // height
    h = remainder % height
    
    # Load slice and normalize
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    
    input_slice = tl.load(input_ptr + 
                         batch * channels * height * width + 
                         channel * height * width + 
                         h * width + offsets, 
                         mask=mask, other=0.0)
    
    sum_val = tl.sum(input_slice)
    normalized_slice = input_slice / (sum_val + 1e-8)
    
    tl.store(out_ptr + 
             batch * channels * height * width + 
             channel * height * width + 
             h * width + offsets, 
             normalized_slice, mask=mask)

@torch.fx.wrap
def optimize_both_branches(in_0, in_1, in_2, in_3):
    # Conv-Sigmoid branch
    batch, in_c, h_in, w_in = in_2.shape
    out_c = in_1.shape[0]
    
    total_conv_elems = batch * 2 * 8 * 8
    BLOCK_SIZE_CONV = 1024
    grid_size_conv = (triton.cdiv(total_conv_elems, BLOCK_SIZE_CONV),)
    
    output_conv = torch.empty((batch, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)
    
    in_1 = in_1.to(in_2.device)
    in_0 = in_0.to(in_2.device)
    
    conv_sigmoid_kernel[grid_size_conv](
        in_1, in_0, in_2, output_conv,
        batch, out_c, in_c, h_in, w_in,
        BLOCK_SIZE=BLOCK_SIZE_CONV
    )
    
    # Sum-Div branch
    batch_sum, channels_sum, height_sum, width_sum = in_3.shape
    num_blocks_sum = batch_sum * channels_sum * height_sum
    BLOCK_SIZE_SUM = min(1024, width_sum)
    grid_size_sum = (num_blocks_sum,)
    
    output_sum = torch.empty_like(in_3)
    
    sum_div_kernel[grid_size_sum](
        in_3, output_sum,
        batch_sum, channels_sum, height_sum, width_sum,
        BLOCK_SIZE=BLOCK_SIZE_SUM
    )
    
    return (output_sum, output_conv)

def replacement_func():
    return optimize_both_branches