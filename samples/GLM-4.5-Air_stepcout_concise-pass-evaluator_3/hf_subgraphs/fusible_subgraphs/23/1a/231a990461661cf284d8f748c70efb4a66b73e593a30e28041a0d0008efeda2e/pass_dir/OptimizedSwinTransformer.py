import torch
import triton
import triton.language as tl

@triton.jit
def swin_conv_kernel_optimized(
    x_ptr,                # Input [1, C, H, W] 
    weight_ptr,           # Weight [C_out, C_in, K, K]
    bias_ptr,             # Bias [C_out]
    out_ptr,              # Output [1, H_out, W_out, C_out]
    batch, channels_in, channels_out, height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    """High-performance Conv2D kernel for Swin Transformer patch embedding"""
    # Program ID for spatial location
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output coordinates
    out_h = pid_h * stride_h
    out_w = pid_w * stride_w
    
    # Early exit if out of bounds
    if out_h >= height or out_w >= width:
        return
    
    # Compute actual window bounds
    win_h = min(kernel_h, height - out_h)
    win_w = min(kernel_w, width - out_w)
    
    # Process channels in blocks
    acc_ptr = out_ptr + (pid_h * width + pid_w) * channels_out + pid_c * BLOCK_SIZE_C
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Load bias for this channel block
    bias_val = 0.0
    if bias_ptr:
        if pid_c * BLOCK_SIZE_C < channels_out:
            bias_val = tl.load(bias_ptr + pid_c * BLOCK_SIZE_C)
    
    # Compute convolution with optimal blocking
    for kh in range(win_h):
        for kw in range(win_w):
            in_h = out_h + kh - pad_h
            in_w = out_w + kw - pad_w
            if 0 <= in_h < height and 0 <= in_w < width:
                # Process input channels in blocks
                for c in range(0, channels_in, BLOCK_SIZE_C):
                    c_end = min(c + BLOCK_SIZE_C, channels_in)
                    
                    # Load weight patch
                    weight_idx = (tl.arange(c_end - c)[:, None] * kernel_w + kw) * channels_in + \
                               tl.broadcast(tl.arange(c_end - c)[:, None], (c_end - c, 1))
                    weight_val = tl.load(weight_ptr + weight_idx, mask=tl.arange(c_end - c) < (c_end - c), other=0.0)
                    
                    # Load input patch
                    input_base = (in_h * width + in_w) * channels_in + c
                    input_val = tl.load(x_ptr + input_base + tl.arange(c_end - c))
                    
                    # Compute dot product
                    for i in range(c_end - c):
                        acc[i] += input_val[i] * weight_val[i]
    
    # Add bias and store
    result = acc + bias_val
    tl.store(acc_ptr, result)
    
    # Store remaining values if needed
    for i in range(BLOCK_SIZE_C):
        if (pid_c * BLOCK_SIZE_C + i) >= channels_out:
            break
        tl.store(out_ptr + (pid_h * width + pid_w) * channels_out + pid_c * BLOCK_SIZE_C + i, 
                result[i])

@torch.fx.wrap  
def optimized_patch_embedding(x, weight, bias):
    """Optimized patch embedding for Swin Transformer"""
    batch, channels_in, height, width = x.shape
    channels_out, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (height - kernel_h) // 4 + 1  # stride 4
    out_w = (width - kernel_w) // 4 + 1
    
    # Create output in BHWN format for efficiency
    output = torch.empty(1, out_h, out_w, channels_out, dtype=x.dtype, device=x.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_C = min(channels_out, 128)
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid dimensions
    grid_h = (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (channels_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    swin_conv_kernel_optimized[(grid_h, grid_w, grid_c)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch=batch, 
        channels_in=channels_in,
        channels_out=channels_out,
        height=height, 
        width=width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=4,
        stride_w=4,
        pad_h=0,
        pad_w=0,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    # Reshape to [1, num_patches, channels] format
    return output.reshape(1, out_h * out_w, channels_out)

def pattern(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    tmp_7 = torch.conv2d(x, weight, bias, (4, 4), (0, 0), (1, 1), 1)
    tmp_8 = tmp_7.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    return tmp_9

def replacement_args(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    return (x, weight, bias)

def replacement_func():
    return optimized_patch_embedding