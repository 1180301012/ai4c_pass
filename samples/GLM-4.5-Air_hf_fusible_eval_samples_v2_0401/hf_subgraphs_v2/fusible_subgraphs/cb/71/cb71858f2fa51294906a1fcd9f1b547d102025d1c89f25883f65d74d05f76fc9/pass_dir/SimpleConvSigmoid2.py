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
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, out_c, in_c, in_h, in_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Output shape will be [batch, 2, 8, 8]
    total_output_elems = batch * 2 * 8 * 8
    mask = pid < total_output_elems
    
    # Map to output coordinates [batch, final_c, final_h, final_w]
    b = pid // (2 * 8 * 8)
    rem = pid % (2 * 8 * 8)
    final_c = rem // (8 * 8)  # 0 or 1
    rem2 = rem % (8 * 8)
    final_h = rem2 // 8      # 0-7
    final_w = rem2 % 8       # 0-7
    
    # Map back to original conv2d output channel [1, 128, 1, 8]
    # 128 total channels → split into 2 groups of 64, each mapped to 8x8 spatial
    channels_per_group = 64
    base_channel_idx = final_c * channels_per_group
    spatial_idx_in_group = final_h * 8 + final_w
    original_channel_idx = base_channel_idx + spatial_idx_in_group
    
    # Compute conv2d result for this specific output position
    conv_val = 0.0
    
    # For output (b, original_channel_idx, 0, final_w), sum over input channels and kernel
    for ic in range(in_c):
        for kh in range(in_h):
            for kw in range(in_w):
                # Load weight: [out_c, in_c, kh, kw] at [original_channel_idx, ic, kh, kw]
                weight_offset = original_channel_idx * in_c * in_h * in_w + ic * in_h * in_w + kh * in_w + kw
                weight = tl.load(weight_ptr + weight_offset, 
                               mask=(original_channel_idx < out_c) & (ic < in_c), 
                               other=0.0)
                
                # Load input: [batch, in_c, in_h, in_w] at [b, ic, kh, final_w]
                # Note: final_w maps to input final_w due to stride=(1,1) and padding=(0,0)
                input_offset = b * in_c * in_h * in_w + ic * in_h * in_w + kh * in_w + final_w
                input_val = tl.load(x_ptr + input_offset, 
                                  mask=(b < batch) & (ic < in_c), 
                                  other=0.0)
                
                conv_val += weight * input_val
    
    # Add bias for this channel
    bias = tl.load(bias_ptr + original_channel_idx, 
                  mask=(original_channel_idx < out_c), 
                  other=0.0)
    conv_val += bias
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Store in [batch, 2, 8, 8] format
    tl.store(out_ptr + pid, sigmoid_val, mask=mask)

@torch.fx.wrap
def simple_conv_sigmoid_func(in_0, in_1, in_2):
    batch, in_c, in_h, in_w = in_2.shape
    out_c = in_1.shape[0]
    
    # Create output tensor in shape [batch, 2, 8, 8]
    output = torch.empty((batch, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)
    
    # Move tensors to same device
    in_1 = in_1.to(in_2.device)
    in_0 = in_0.to(in_2.device)
    
    # Launch kernel
    total_elems = batch * 2 * 8 * 8
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elems, BLOCK_SIZE),)
    
    simple_conv_sigmoid_kernel[grid_size](
        in_2, in_1, in_0, output,
        batch, out_c, in_c, in_h, in_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_conv_sigmoid_func