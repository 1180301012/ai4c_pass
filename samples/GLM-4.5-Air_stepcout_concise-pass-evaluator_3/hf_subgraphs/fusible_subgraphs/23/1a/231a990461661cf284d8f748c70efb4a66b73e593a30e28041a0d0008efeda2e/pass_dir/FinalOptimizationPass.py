import torch
import triton
import triton.language as tl

@triton.jit
def optimized_conv_transpose_kernel(
    x_ptr,                # Input [1, C, H, W]
    weight_ptr,           # Weight [C_out, C_in, 4, 4]
    bias_ptr,             # Bias [C_out]
    out_ptr,              # Output [1, H_out, W_out, C_out]
    batch_in, channels_in, height_in, width_in,
    channels_out, height_out, width_out,
    BLOCK_SIZE_N: tl.constexpr
):
    """Ultimate optimized Conv2D + Flatten + Transpose fusion kernel"""
    pid = tl.program_id(0)
    
    if pid >= height_out * width_out:
        return
    
    # Calculate output coordinates
    out_h = pid // width_out
    out_w = pid % width_out
    
    # Calculate input coordinates (with stride 4)
    in_h = out_h * 4
    in_w = out_w * 4
    
    # Output pointer
    out_base = out_ptr + pid * channels_out
    
    # Initialize accumulator
    acc = tl.zeros((channels_out,), dtype=tl.float32, device=tl.device.current())
    
    # Process 4x4 kernel
    for kh in range(4):
        for kw in range(4):
            if in_h + kh < height_in and in_w + kw < width_in:
                # Load input patch
                offset = (in_h + kh) * width_in + in_w + kw
                x_val = tl.load(x_ptr + offset * channels_in)
                
                # Load corresponding weights (vectorized)
                weight_offset = kh * 4 + kw
                for c_out in range(0, channels_out, BLOCK_SIZE_N):
                    c_end = min(c_out + BLOCK_SIZE_N, channels_out)
                    weights = tl.load(weight_ptr + offset * channels_out + 
                                    tl.arange(c_out, c_end))
                    x_slice = x_val[tl.arange(c_out, c_end)]
                    
                    # Vectorized dot product
                    acc[c_out:c_end] += x_slice * weights
    
    # Load bias if available
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(channels_out))
        acc += bias
    
    # Store result
    tl.store(out_base, acc)

@torch.fx.wrap
def ultimate_optimized_conv_embed(x, weight, bias):
    """Ultimate optimized patch embedding for Swin Transformer"""
    batch, channels, height, width = x.shape
    channels_out, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions with stride 4
    out_h = (height - kernel_h) // 4 + 1
    out_w = (width - kernel_w) // 4 + 1
    total_patches = out_h * out_w
    
    # Create output directly in final format [1, total_patches, channels_out]
    output = torch.empty(1, total_patches, channels_out, 
                        dtype=x.dtype, device=x.device)
    
    # Optimal block size for maximum occupancy on A30
    BLOCK_SIZE_N = 64
    grid_size = (total_patches,)
    
    # Launch kernel with optimal configuration
    optimized_conv_transpose_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_in=batch,
        channels_in=channels,
        height_in=height,
        width_in=width,
        channels_out=channels_out,
        height_out=out_h,
        width_out=out_w,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def pattern(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    fused_conv = torch.conv2d(x, weight, bias, (4, 4), (0, 0), (1, 1), 1)
    patches = fused_conv.flatten(2)
    embedded = patches.transpose(1, 2)
    return embedded

def replacement_args(x, weight, bias, normalized_shape, weight_norm, bias_norm, eps, dropout_p, dropout_training, dropout_inplace):
    return (x, weight, bias)

def replacement_func():
    return ultimate_optimized_conv_embed