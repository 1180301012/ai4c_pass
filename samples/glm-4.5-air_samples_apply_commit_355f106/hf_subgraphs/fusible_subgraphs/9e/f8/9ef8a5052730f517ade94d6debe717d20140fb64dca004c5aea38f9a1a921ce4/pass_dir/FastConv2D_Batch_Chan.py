import torch
import triton
import triton.language as tl

@triton.jit
def fast_conv2d_kernel(
    x_ptr,  # Input tensor [batch_size, in_channels, H, W]
    weight_ptr,  # Weight tensor [out_channels, in_channels, KH, KW]
    out_ptr,  # Output tensor [batch_size, out_channels, H, W]
    batch_size, in_channels, out_channels, height, width, kh, kw,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # Output channel
    pid_n = tl.program_id(1)  # Batch index
    
    out_h = (height + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    
    m_mask = pid_m < out_channels
    n_mask = pid_n < batch_size
    
    # Use scalar accumulator to avoid vectorization issues
    accum = 0.0
    
    # Process one weight at a time to keep things simple
    for k in range(in_channels * kh * kw):
        in_c = k // (kh * kw)
        kh_idx = (k % (kh * kw)) // kw
        kw_idx = (k % (kh * kw)) % kw
        
        in_h = pid_m * stride_h - pad_h + kh_idx * dilation_h
        in_w = pid_n * stride_w - pad_w + kw_idx * dilation_w
        
        # Check bounds
        if (in_c < in_channels and kh_idx < kh and kw_idx < kw) and \
           (in_h >= 0 and in_h < height and in_w >= 0 and in_w < width) and \
           (m_mask and n_mask):
            
            weight_offset = pid_m * in_channels * kh * kw + k
            weight = tl.load(weight_ptr + weight_offset, other=0.0)
            
            in_offset = pid_n * in_channels * height * width + in_c * height * width + in_h * width + in_w
            in_val = tl.load(x_ptr + in_offset, other=0.0)
            
            accum += weight.item() * in_val.item()
    
    out_offset = pid_n * out_channels * out_h * out_w + pid_m * out_h * out_w
    tl.store(out_ptr + out_offset, accum, mask=m_mask & n_mask)

@torch.fx.wrap
def fast_conv2d_optimized(x, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    batch_size, in_channels, height, width = x.shape
    out_channels, weight_in_channels, kh, kw = weight.shape
    
    
    
    # Calculate output dimensions more carefully
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
        
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    out_h = (height + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    
    # Sanity check to prevent negative dimensions
    if out_h <= 0 or out_w <= 0:
        raise RuntimeError(f"Invalid output dimensions: {out_h}x{out_w} for input {height}x{width} with kernel {kh}x{kw}, padding {pad_h}x{pad_w}, stride {stride_h}x{stride_w}")
    
    out = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 32
    
    grid = (
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    fast_conv2d_kernel[grid](
        x_ptr=x, weight_ptr=weight, out_ptr=out,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        height=height, width=width, kh=kh, kw=kw,
        stride_h=stride, stride_w=stride, pad_h=padding, pad_w=padding,
        dilation_h=dilation, dilation_w=dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def pattern(conv_input, conv_weight):
    # Match just the core convolution operation
    # Note: pytorch.conv2d expects (input, weight, ...) so this matches
    conv_result = torch.conv2d(conv_input, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    return conv_result

def replacement_args(conv_input, conv_weight):
    return (conv_input, conv_weight)

def replacement_func():
    return fast_conv2d_optimized