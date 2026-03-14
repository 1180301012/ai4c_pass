import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.cat((tmp_1, in_2), 1)
    return (tmp_2,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_kernel(
    input_ptr,      # [batch, in_channels, height, width]
    weight_ptr,     # [out_channels, in_channels, kernel_h, kernel_w]
    out_ptr,        # [batch, out_channels, height, width]
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    batch = pid_m // out_channels
    channel = pid_m % out_channels
    
    if batch >= batch_size:
        h = 0
        w = 0
    else:
        h = pid_n // out_width
        w = pid_n % out_width
        if h >= out_height or w >= out_width:
            tl.store(out_ptr + batch * out_channels * out_height * out_width + channel * out_height * out_width + h * out_width + w, 0.0)
            return
    
    if channel >= out_channels:
        tl.store(out_ptr, 0.0)
        return
    
    val = tl.zeros([], dtype=tl.float32)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h * stride_h - pad_h + kh * dilation_h
            w_in = w * stride_w - pad_w + kw * dilation_w
            
            if (0 <= h_in):
                if (h_in < height):
                    if (0 <= w_in):
                        if (w_in < width):
                            for k in range(in_channels // groups):
                                in_ch = (channel % groups) * (in_channels // groups) + k
                                offset_in = batch * in_channels * height * width + in_ch * height * width + h_in * width + w_in
                                offset_w = channel * in_channels * kernel_h * kernel_w + in_ch * kernel_h * kernel_w + kh * kernel_w + kw
                                
                                input_val = tl.load(input_ptr + offset_in, mask=None, boundary_check=False)
                                weight_val = tl.load(weight_ptr + offset_w, mask=None, boundary_check=False)
                                val += input_val * weight_val
    
    out_offset = batch * out_channels * out_height * out_width + channel * out_height * out_width + h * out_width + w
    tl.store(out_ptr + out_offset, val)

@triton.jit
def concat_kernel(
    conv_ptr,       # [batch, conv_channels, height, width]
    concat_ptr,     # [batch, concat_channels, height, width]
    out_ptr,        # [batch, conv_channels + concat_channels, height, width]
    conv_channels,
    concat_channels,
    height,
    width,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_channels = conv_channels + concat_channels
    
    if pid >= batch_size * height * width * total_channels:
        return
    
    batch = pid // (height * width * total_channels)
    channel = (pid % (height * width * total_channels)) // (height * width)
    h = (pid % (height * width)) // width
    w = pid % width
    
    if channel < conv_channels:
        offset = batch * conv_channels * height * width + channel * height * width + h * width + w
        tl.store(out_ptr + pid, tl.load(conv_ptr + offset, mask=None, boundary_check=False))
    else:
        concat_channel = channel - conv_channels
        offset = batch * concat_channels * height * width + concat_channel * height * width + h * width + w
        tl.store(out_ptr + pid, tl.load(concat_ptr + offset, mask=None, boundary_check=False))

@torch.fx.wrap
def optimized_conv2d_concat(in_0, in_1, in_2):
    B, C_in, H, W = in_1.shape
    K, _, KH, KW = in_0.shape
    C_concat = in_2.shape[1]
    
    out = torch.empty((B, K + C_concat, H, W), device=in_1.device, dtype=in_1.dtype)
    
    # Perform convolution
    conv_out = torch.empty((B, K, H, W), device=in_1.device, dtype=in_1.dtype)
    
    if B > 0 and K > 0 and H > 0 and W > 0:
        grid_conv = (
            triton.cdiv(B * K, 64),   # batch * channels (better for occupancy)
            triton.cdiv(H * W, 256)   # spatial dimensions (better for memory coalescing)
        )
        
        conv2d_kernel[grid_conv](
            in_1,
            in_0,
            conv_out,
            B,
            C_in,
            H,
            W,
            K,
            KH,
            KW,
            1, 1,  # stride_h, stride_w
            1, 1,  # pad_h, pad_w  
            1, 1,  # dilation_h, dilation_w
            1,     # groups
            64,    # BLOCK_SIZE_M (optimized for better GPU occupancy)
            256    # BLOCK_SIZE_N (optimized for memory coalescing)
        )
    
    # Perform concatenation
    total_elements = B * (K + C_concat) * H * W
    if total_elements > 0:
        grid_concat = (triton.cdiv(total_elements, 2048),)  # Larger block size for better efficiency
        
        concat_kernel[grid_concat](
            conv_out,
            in_2,
            out,
            K,
            C_concat,
            H,
            W,
            B,
            2048  # Optimized block size
        )
    
    return out

def replacement_func():
    return optimized_conv2d_concat