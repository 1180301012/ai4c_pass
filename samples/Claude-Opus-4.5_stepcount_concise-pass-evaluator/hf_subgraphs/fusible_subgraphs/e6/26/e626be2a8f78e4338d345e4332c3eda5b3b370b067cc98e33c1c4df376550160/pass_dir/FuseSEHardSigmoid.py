import torch
import triton
import triton.language as tl

# Pattern matching function - matches post-conv Hard Sigmoid + broadcast multiply
def pattern(conv_out, x):
    """
    Match the post-conv pattern:
    conv_out + 1.0 -> div 2.0 -> clamp_(0, 1) -> multiply with x (with broadcasting)
    
    conv_out shape: [batch, channels, 1, 1]
    x shape: [batch, channels, H, W]
    """
    tmp_3 = conv_out + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = x * tmp_5
    return tmp_6


def replacement_args(conv_out, x):
    return (conv_out, x)


@triton.jit
def fused_hardsigmoid_mul_kernel_2d(
    conv_out_ptr,
    x_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Hard Sigmoid + broadcast multiply.
    2D grid: (batch * channels, ceil(HW / BLOCK_SIZE))
    """
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    # Load conv output value for this (batch, channel)
    conv_val = tl.load(conv_out_ptr + pid_bc)
    
    # Hard sigmoid: (x + 1) / 2, clamped to [0, 1]
    sigmoid_val = (conv_val + 1.0) * 0.5
    sigmoid_val = tl.maximum(sigmoid_val, 0.0)
    sigmoid_val = tl.minimum(sigmoid_val, 1.0)
    
    # Calculate offsets
    base_offset = pid_bc * HW
    hw_start = pid_hw * BLOCK_SIZE
    offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    
    # Load x, multiply with sigmoid, store
    x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask)
    out_vals = x_vals * sigmoid_val
    tl.store(out_ptr + base_offset + offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_hardsigmoid_mul(conv_out, x):
    """
    Optimized Hard Sigmoid + broadcast multiply.
    
    conv_out shape: [batch, channels, 1, 1]
    x shape: [batch, channels, H, W]
    """
    batch = x.shape[0]
    channels = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HW = H * W
    batch_channels = batch * channels
    
    # Flatten conv_out
    conv_out_flat = conv_out.view(batch_channels)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Use block size 128 with num_warps=2
    BLOCK_SIZE = 128
    
    # 2D grid
    grid = (batch_channels, triton.cdiv(HW, BLOCK_SIZE))
    
    fused_hardsigmoid_mul_kernel_2d[grid](
        conv_out_flat,
        x,
        out,
        HW,
        BLOCK_SIZE,
        num_warps=2,
    )
    
    return out


def replacement_func():
    return fused_hardsigmoid_mul