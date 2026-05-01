import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_8, in_2, in_1, in_0):
    conv = torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1)
    drop = torch.nn.functional.dropout(conv, 0.0, False, False)
    return drop * in_0

# Argument extraction function
def replacement_args(in_8, in_2, in_1, in_0):
    return (in_8, in_2, in_1, in_0)

# Triton kernel for fused convolution and multiplication
@triton.jit
def fused_conv_mul_kernel(
    in_ptr, w_ptr, b_ptr, scale_ptr, out_ptr,
    batch, H, W, in_channels, out_channels,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    start_h = pid_h * BLOCK_SIZE_H
    start_w = pid_w * BLOCK_SIZE_W
    start_c = pid_c * BLOCK_SIZE_C
    
    # Load scale and bias for the channel block
    # Load scale and bias for the channel block directly
    for c in range(start_c, min(start_c + BLOCK_SIZE_C, out_channels)):

        scale_val = tl.load(scale_ptr + c)
        bias_val = tl.load(b_ptr + c)
        
        acc = 0.0
        for k in range(in_channels):
            # Calculate input pointer
            input_idx = (start_h * W + start_w) * in_channels + k
            input_val = tl.load(in_ptr + input_idx)
            weight_val = tl.load(w_ptr + c * in_channels + k)
            acc += input_val * weight_val
        acc += bias_val
        acc *= scale_val
        out_idx = (start_h * W + start_w) * out_channels + c
        tl.store(out_ptr + out_idx, acc)

# Kernel wrapper
@torch.fx.wrap
def fused_conv_mul(in_8, in_2, in_1, in_0):
    batch, in_channels, H, W = in_8.shape
    out_channels = in_2.shape[0]  # Shape [out_channels, in_channels, 1, 1]
    
    out = torch.empty(batch, out_channels, H, W, dtype=in_8.dtype, device=in_8.device)
    
    # Block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 16
    
    num_blocks_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_blocks_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    fused_conv_mul_kernel[(num_blocks_h, num_blocks_w, num_blocks_c)](
        in_8,
        in_2,
        in_1,
        in_0,
        out,
        batch,
        H,
        W,
        in_channels,
        out_channels,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_mul