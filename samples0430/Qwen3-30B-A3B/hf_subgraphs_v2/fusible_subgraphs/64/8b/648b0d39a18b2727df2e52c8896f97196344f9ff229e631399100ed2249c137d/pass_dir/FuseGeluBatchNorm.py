import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_4, in_5, in_0, in_1, in_2, in_3):
    in_6 = in_4 + in_5
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5, tmp_6

# Argument extraction function
def replacement_args(in_4, in_5, in_0, in_1, in_2, in_3):
    return (in_4, in_5, in_0, in_1, in_2, in_3)

# Triton kernel for fused Gelu + BatchNorm
@triton.jit
def fused_gelu_bn_kernel(
    in4_ptr, in5_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    gelu_out_ptr, bn_out_ptr,
    batch, channels, height, width,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Calculate block index
    block_id = tl.program_id(0)
    num_h = (height + BLOCK_H - 1) // BLOCK_H
    num_w = (width + BLOCK_W - 1) // BLOCK_W
    total_blocks = batch * num_h * num_w

    if block_id >= total_blocks:
        return

    batch_id = block_id // (num_h * num_w)
    block_id_in_h_w = block_id % (num_h * num_w)
    h_id = block_id_in_h_w // num_w
    w_id = block_id_in_h_w % num_w

    # Compute spatial coordinates
    h = h_id * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    w = w_id * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    c = tl.arange(0, BLOCK_C)

    # Mask for valid spatial and channel indices
    mask_h = (h < height)[:, None]
    mask_w = (w < width)[None, :]
    mask_c = (c < channels)[None, :]
    mask = mask_h & mask_w & mask_c

    # Load input tensors
    in4 = tl.load(
        in4_ptr + (batch_id * channels * height * width + 
                 c * height * width + h * width + w),
        mask=mask,
        other=0.0
    )
    in5 = tl.load(
        in5_ptr + (batch_id * channels * height * width + 
                 c * height * width + h * width + w),
        mask=mask,
        other=0.0
    )
    x = in4 + in5

    # Compute Gelu: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    sqrt2_pi = 0.7978845608
    c1 = 0.044715
    x2 = x * x
    x_scaled = x * (1 + c1 * x2)
    tanh_val = tl.tanh(sqrt2_pi * x_scaled)
    gelu_x = 0.5 * x * (1 + tanh_val)

    # Load BatchNorm parameters
    mean = tl.load(mean_ptr + c, mask=mask_c)
    var = tl.load(var_ptr + c, mask=mask_c)
    weight = tl.load(weight_ptr + c, mask=mask_c)
    bias = tl.load(bias_ptr + c, mask=mask_c)

    # Apply BatchNorm
    denom = tl.sqrt(var + 1e-5)
    x_norm = (gelu_x - mean) / denom
    bn_out = weight * x_norm + bias

    # Store results
    tl.store(gelu_out_ptr + (batch_id * channels * height * width + 
                            c * height * width + h * width + w), 
             gelu_x, mask=mask)
    tl.store(bn_out_ptr + (batch_id * channels * height * width + 
                          c * height * width + h * width + w), 
             bn_out, mask=mask)

# Wrapper for the Triton kernel
@torch.fx.wrap
def fused_gelu_bn(in_4, in_5, in_0, in_1, in_2, in_3):
    batch, channels, height, width = in_4.shape
    
    # Constants for kernel tiling
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_C = 32
    
    # Allocate output tensors
    gelu_out = torch.empty_like(in_4)
    bn_out = torch.empty_like(in_4)

    # Calculate grid dimensions
    num_h = (height + BLOCK_H - 1) // BLOCK_H
    num_w = (width + BLOCK_W - 1) // BLOCK_W
    total_blocks = batch * num_h * num_w

    # Launch kernel
    fused_gelu_bn_kernel[(total_blocks,)](
        in_4, in_5, in_0, in_1, in_2, in_3,
        gelu_out, bn_out,
        batch, channels, height, width,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C
    )
    
    return gelu_out, bn_out

# Replacement function
@torch.fx.wrap
def replacement_func():
    return fused_gelu_bn