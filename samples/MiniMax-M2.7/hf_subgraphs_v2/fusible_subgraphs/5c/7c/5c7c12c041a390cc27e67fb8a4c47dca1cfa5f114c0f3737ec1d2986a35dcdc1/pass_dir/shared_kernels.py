"""
Shared Triton kernels for all ERFNet encoder-decoder fusion passes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_16channels(output_ptr, in_4_ptr, in_5_ptr, 
                            mean_ptr, var_ptr, weight_ptr, bias_ptr,
                            bn_eps: tl.constexpr,
                            output_height: tl.constexpr, output_width: tl.constexpr,
                            n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Fused kernel for 16-channel batch normalization (start1_end6_0 pattern).
    Input: [B, 3, 512, 512] -> max_pool2d -> [B, 3, 256, 256] -> interpolate -> [B, 3, 256, 256]
    Concat with: [B, 13, 256, 256] -> [B, 16, 256, 256] -> BN -> ReLU
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    channels = 16
    height = output_height  # 256
    width = output_width    # 256
    
    # Output layout: [B, 16, 256, 256]
    out_c = (offsets % (channels * height * width)) // (height * width)
    out_spatial = offsets % (height * width)
    out_h = out_spatial // width
    out_w = out_spatial % width
    
    # Batch index for accessing in_5
    b = offsets // (channels * height * width)
    
    # in_4: [B, 13, 256, 256]
    # Layout: b*13*256*256 + c*256*256 + h*256 + w
    in_4_c = 13
    in_4_base = b * in_4_c * height * width
    in_4_offset = out_c * height * width + out_h * width + out_w
    
    # in_5: [B, 3, 512, 512]
    # After max_pool2d(2,2): [B, 3, 256, 256]
    # For max pooling, we need max of 2x2 block
    # in_5 input is [B, 3, 512, 512]
    # pool_h = out_h, pool_w = out_w maps to input at (2*h, 2*w), (2*h+1, 2*w), etc.
    in_5_c = 3
    
    # For channels 0-12 (in_4): direct load
    # For channels 13-15 (in_5): max_pool2d + interpolate (256->256 is identity)
    # Load in_4 for channels 0-12
    val_in_4 = tl.load(in_4_ptr + in_4_base + in_4_offset, mask=mask, other=0.0)
    
    # For in_5, we need max_pool2d: load 4 values from 2x2 block and take max
    # in_5 is [B, 3, 512, 512]
    # After max_pool2d(2,2): [B, 3, 256, 256]
    # Pooled at (c, h, w) = max(input[c, 2*h, 2*w], input[c, 2*h+1, 2*w], input[c, 2*h, 2*w+1], input[c, 2*h+1, 2*w+1])
    in_5_base_h = out_h * 2
    in_5_base_w = out_w * 2
    c_in = out_c - 13  # Channel within in_5 (0-2)
    
    # Need to handle bounds checking for max_pool2d (last row/col might exceed)
    # For simplicity, we clamp to valid range
    in_5_offset_base = b * in_5_c * 512 * 512
    
    # Load 4 corners for max pooling
    p00 = in_5_offset_base + c_in * 512 * 512 + in_5_base_h * 512 + in_5_base_w
    p01 = in_5_offset_base + c_in * 512 * 512 + in_5_base_h * 512 + (in_5_base_w + 1)
    p10 = in_5_offset_base + c_in * 512 * 512 + (in_5_base_h + 1) * 512 + in_5_base_w
    p11 = in_5_offset_base + c_in * 512 * 512 + (in_5_base_h + 1) * 512 + (in_5_base_w + 1)
    
    v00 = tl.load(in_5_ptr + p00, mask=mask, other=0.0)
    v01 = tl.load(in_5_ptr + p01, mask=mask, other=0.0)
    v10 = tl.load(in_5_ptr + p10, mask=mask, other=0.0)
    v11 = tl.load(in_5_ptr + p11, mask=mask, other=0.0)
    
    val_in_5 = tl.max(tl.max(tl.max(v00, v01), v10), v11)
    
    # Select: in_4 for channels 0-12, in_5 for channels 13-15
    val = tl.where(out_c < 13, val_in_4, val_in_5)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    mean_val = tl.load(mean_ptr + out_c)
    var_val = tl.load(var_ptr + out_c)
    weight_val = tl.load(weight_ptr + out_c)
    bias_val = tl.load(bias_ptr + out_c)
    
    std = tl.sqrt(var_val + bn_eps)
    val = (val - mean_val) / std * weight_val + bias_val
    
    # ReLU
    val = tl.maximum(val, 0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)


@triton.jit
def fused_kernel_64channels(output_ptr, in_4_ptr, in_5_ptr,
                            mean_ptr, var_ptr, weight_ptr, bias_ptr,
                            bn_eps: tl.constexpr,
                            output_height: tl.constexpr, output_width: tl.constexpr,
                            n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Fused kernel for 64-channel batch normalization (start7_end12_1 pattern).
    Input: in_5 [B, 16, 256, 256] -> max_pool2d -> [B, 16, 128, 128] -> interpolate(128,128) -> [B, 16, 128, 128]
    Concat with: in_4 [B, 48, 128, 128] -> [B, 64, 128, 128] -> BN -> ReLU
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    channels = 64
    height = output_height  # 128
    width = output_width    # 128
    
    # Output layout: [B, 64, 128, 128]
    out_c = (offsets % (channels * height * width)) // (height * width)
    out_spatial = offsets % (height * width)
    out_h = out_spatial // width
    out_w = out_spatial % width
    
    b = offsets // (channels * height * width)
    
    # in_4: [B, 48, 128, 128]
    in_4_c = 48
    in_4_base = b * in_4_c * height * width
    in_4_offset = out_c * height * width + out_h * width + out_w
    val_in_4 = tl.load(in_4_ptr + in_4_base + in_4_offset, mask=mask, other=0.0)
    
    # in_5: [B, 16, 256, 256] -> max_pool2d(2,2) -> [B, 16, 128, 128] -> interpolate(128,128) -> [B, 16, 128, 128]
    # For max_pool2d: pooled[b, c, h, w] = max(input[b, c, 2*h, 2*w], input[b, c, 2*h+1, 2*w], ...)
    in_5_c = 16
    c_in = out_c - 48  # Channel within in_5 (0-15)
    
    in_5_base_h = out_h * 2
    in_5_base_w = out_w * 2
    in_5_offset_base = b * in_5_c * 256 * 256
    
    # Load 4 corners for max pooling
    p00 = in_5_offset_base + c_in * 256 * 256 + in_5_base_h * 256 + in_5_base_w
    p01 = in_5_offset_base + c_in * 256 * 256 + in_5_base_h * 256 + (in_5_base_w + 1)
    p10 = in_5_offset_base + c_in * 256 * 256 + (in_5_base_h + 1) * 256 + in_5_base_w
    p11 = in_5_offset_base + c_in * 256 * 256 + (in_5_base_h + 1) * 256 + (in_5_base_w + 1)
    
    v00 = tl.load(in_5_ptr + p00, mask=mask, other=0.0)
    v01 = tl.load(in_5_ptr + p01, mask=mask, other=0.0)
    v10 = tl.load(in_5_ptr + p10, mask=mask, other=0.0)
    v11 = tl.load(in_5_ptr + p11, mask=mask, other=0.0)
    
    val_in_5 = tl.max(tl.max(tl.max(v00, v01), v10), v11)
    
    # Select: in_4 for channels 0-47, in_5 for channels 48-63
    val = tl.where(out_c < 48, val_in_4, val_in_5)
    
    # BatchNorm
    mean_val = tl.load(mean_ptr + out_c)
    var_val = tl.load(var_ptr + out_c)
    weight_val = tl.load(weight_ptr + out_c)
    bias_val = tl.load(bias_ptr + out_c)
    
    std = tl.sqrt(var_val + bn_eps)
    val = (val - mean_val) / std * weight_val + bias_val
    
    # ReLU
    val = tl.maximum(val, 0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)


@triton.jit
def fused_kernel_128channels(output_ptr, in_4_ptr, in_5_ptr,
                             mean_ptr, var_ptr, weight_ptr, bias_ptr,
                             bn_eps: tl.constexpr,
                             output_height: tl.constexpr, output_width: tl.constexpr,
                             n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Fused kernel for 128-channel batch normalization (start73_end78_8 pattern).
    Input: in_5 [B, 64, 128, 128] -> max_pool2d -> [B, 64, 64, 64] -> interpolate(64,64) -> [B, 64, 64, 64]
    Concat with: in_4 [B, 64, 64, 64] -> [B, 128, 64, 64] -> BN -> ReLU
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    channels = 128
    height = output_height  # 64
    width = output_width    # 64
    
    # Output layout: [B, 128, 64, 64]
    out_c = (offsets % (channels * height * width)) // (height * width)
    out_spatial = offsets % (height * width)
    out_h = out_spatial // width
    out_w = out_spatial % width
    
    b = offsets // (channels * height * width)
    
    # in_4: [B, 64, 64, 64]
    in_4_c = 64
    in_4_base = b * in_4_c * height * width
    in_4_offset = out_c * height * width + out_h * width + out_w
    val_in_4 = tl.load(in_4_ptr + in_4_base + in_4_offset, mask=mask, other=0.0)
    
    # in_5: [B, 64, 128, 128] -> max_pool2d(2,2) -> [B, 64, 64, 64] -> interpolate(64,64) -> [B, 64, 64, 64]
    in_5_c = 64
    c_in = out_c - 64  # Channel within in_5 (0-63)
    
    in_5_base_h = out_h * 2
    in_5_base_w = out_w * 2
    in_5_offset_base = b * in_5_c * 128 * 128
    
    # Load 4 corners for max pooling
    p00 = in_5_offset_base + c_in * 128 * 128 + in_5_base_h * 128 + in_5_base_w
    p01 = in_5_offset_base + c_in * 128 * 128 + in_5_base_h * 128 + (in_5_base_w + 1)
    p10 = in_5_offset_base + c_in * 128 * 128 + (in_5_base_h + 1) * 128 + in_5_base_w
    p11 = in_5_offset_base + c_in * 128 * 128 + (in_5_base_h + 1) * 128 + (in_5_base_w + 1)
    
    v00 = tl.load(in_5_ptr + p00, mask=mask, other=0.0)
    v01 = tl.load(in_5_ptr + p01, mask=mask, other=0.0)
    v10 = tl.load(in_5_ptr + p10, mask=mask, other=0.0)
    v11 = tl.load(in_5_ptr + p11, mask=mask, other=0.0)
    
    val_in_5 = tl.max(tl.max(tl.max(v00, v01), v10), v11)
    
    # Select: in_4 for channels 0-63, in_5 for channels 64-127
    val = tl.where(out_c < 64, val_in_4, val_in_5)
    
    # BatchNorm
    mean_val = tl.load(mean_ptr + out_c)
    var_val = tl.load(var_ptr + out_c)
    weight_val = tl.load(weight_ptr + out_c)
    bias_val = tl.load(bias_ptr + out_c)
    
    std = tl.sqrt(var_val + bn_eps)
    val = (val - mean_val) / std * weight_val + bias_val
    
    # ReLU
    val = tl.maximum(val, 0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)


# ============================================================================
# Dispatch Wrapper
# ============================================================================

@torch.fx.wrap
def fused_dispatch(in_4, in_5, mean, var, weight, bias, route):
    """
    Dispatch wrapper that routes to the appropriate kernel based on configuration.
    """
    B, C_cat, H, W = in_4.shape
    
    # Determine output shape based on route
    if route == "16ch":
        output_channels = 16
        output_height = 256
        output_width = 256
    elif route == "64ch":
        output_channels = 64
        output_height = 128
        output_width = 128
    elif route == "128ch":
        output_channels = 128
        output_height = 64
        output_width = 64
    else:
        raise ValueError(f"Unknown route: {route}")
    
    N = B * output_channels * output_height * output_width
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((B, output_channels, output_height, output_width), 
                         dtype=in_4.dtype, device=in_4.device)
    
    bn_eps = 0.001
    
    # Convert to contiguous if needed
    in_4_cont = in_4.contiguous()
    in_5_cont = in_5.contiguous()
    
    if route == "16ch":
        fused_kernel_16channels[(num_programs,)](
            output, in_4_cont, in_5_cont,
            mean, var, weight, bias,
            bn_eps, output_height, output_width,
            N, BLOCK_SIZE
        )
    elif route == "64ch":
        fused_kernel_64channels[(num_programs,)](
            output, in_4_cont, in_5_cont,
            mean, var, weight, bias,
            bn_eps, output_height, output_width,
            N, BLOCK_SIZE
        )
    elif route == "128ch":
        fused_kernel_128channels[(num_programs,)](
            output, in_4_cont, in_5_cont,
            mean, var, weight, bias,
            bn_eps, output_height, output_width,
            N, BLOCK_SIZE
        )
    
    return output