import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_flatten_transpose_kernel(
    x_ptr,                # Input tensor [1, 3, 384, 384]
    weight_ptr,           # Weight [128, 3, 4, 4] 
    bias_ptr,             # Bias [128]
    out_ptr,              # Output [1, 96*96, 128]
    batch, channels, height, width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program ID for output patches
    pid_m = tl.program_id(0)  # Flat patch index
    num_patches = height // stride_h * (width // stride_w)
    
    if pid_m >= num_patches:
        return
    
    # Calculate output coordinates from flat index
    out_h = (pid_m // (width // stride_w)) * stride_h
    out_w = (pid_m % (width // stride_w)) * stride_w
    
    # Load bias
    bias = tl.load(bias_ptr + pid_m % (width // stride_w))
    
    # Iterate over input channels
    acc = 0.0
    for c in range(0, channels, BLOCK_SIZE_K):
        # Load weight patch
        weight_offsets = c + tl.arange(0, BLOCK_SIZE_K)
        weight_mask = weight_offsets < channels
        weight_block = tl.load(weight_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * kernel_w * kernel_w + 
                                           tl.arange(0, kernel_w)[None, :] * kernel_w + 
                                           tl.arange(0, kernel_w)[None, :]), 
                             mask=weight_mask[None, :], other=0.0)
        
        # Load input patch for this channel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_h = out_h + kh - pad_h
                in_w = out_w + kw - pad_w
                if 0 <= in_h < height and 0 <= in_w < width:
                    input_base = (in_h * width + in_w) * channels + c
                    input_offsets = input_base + tl.arange(0, BLOCK_SIZE_K)
                    input_mask = input_offsets < (height * width * channels)
                    input_block = tl.load(x_ptr + input_offsets, mask=input_mask, other=0.0)
                    
                    # Compute dot product
                    acc += tl.sum(input_block * weight_block[kh, kw, :])
    
    # Store result
    tl.store(out_ptr + pid_m * out_channels + tl.arange(0, out_channels), acc, mask=tl.arange(0, out_channels) < out_channels)

@torch.fx.wrap
def fused_conv2d_flatten_transpose(x, weight, bias):
    # Input shapes
    batch, channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions for conv2d with stride 4
    out_h = (height + 2*0 - kernel_h) // 4 + 1  # = 96
    out_w = (width + 2*0 - kernel_w) // 4 + 1   # = 96
    
    # Calculate total number of patches after flatten and transpose
    num_patches = out_h * out_w
    
    # Create output tensor [1, num_patches, out_channels]
    out = torch.empty(1, num_patches, out_channels, dtype=x.dtype, device=x.device)
    
    # Triton kernel launch parameters
    BLOCK_SIZE_M = 1  # Each program handles one patch
    BLOCK_SIZE_N = out_channels
    BLOCK_SIZE_K = min(32, channels)  # Channel block size
    
    grid = (num_patches,)
    
    fused_conv2d_flatten_transpose_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch=batch, channels=channels, height=height, width=width,
        out_channels=out_channels, kernel_h=kernel_h, kernel_w=kernel_w,
        stride_h=4, stride_w=4, pad_h=0, pad_w=0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def pattern(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    tmp_7 = torch.conv2d(x, weight, bias, (4, 4), (0, 0), (1, 1), 1)
    tmp_8 = tmp_7.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    return tmp_9

def replacement_args(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    return (x, weight, bias)

def replacement_func():
    return fused_conv2d_flatten_transpose