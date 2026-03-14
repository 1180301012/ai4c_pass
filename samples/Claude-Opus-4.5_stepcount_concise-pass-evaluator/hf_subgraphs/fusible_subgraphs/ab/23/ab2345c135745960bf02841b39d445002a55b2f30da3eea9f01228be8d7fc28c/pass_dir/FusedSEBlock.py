import torch
import triton
import triton.language as tl

# Pattern to match: conv2d -> sigmoid -> view -> multiply -> contiguous
def pattern(bias, weight, x, x_gap):
    conv_out = torch.conv2d(x_gap, weight, bias, (1, 1), (0, 0), (1, 1), 4)
    sig_out = torch.sigmoid(conv_out)
    viewed = sig_out.view(1, -1, 1, 1)
    mul_out = x * viewed
    result = mul_out.contiguous()
    return result

def replacement_args(bias, weight, x, x_gap):
    return (bias, weight, x, x_gap)


@triton.jit
def fused_se_kernel(
    bias_ptr, weight_ptr, x_ptr, x_gap_ptr, out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    # 2D grid: (channel, hw_block)
    c_idx = tl.program_id(0)  # channel index [0, 96)
    hw_block = tl.program_id(1)  # spatial block
    
    hw_start = hw_block * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW
    
    # Compute attention weight for this channel (grouped conv2d)
    # groups=4, in_channels=32, out_channels=96
    # Each group: 8 input channels -> 24 output channels
    group = c_idx // 24
    
    # Accumulate: bias + weight @ x_gap
    acc = tl.load(bias_ptr + c_idx).to(tl.float32)
    
    # Load all 8 weights for this output channel using vectorized load
    weight_base = c_idx * 8
    w = tl.load(weight_ptr + weight_base + tl.arange(0, 8))
    
    # Load all 8 input values for this group
    base_in = group * 8
    xg = tl.load(x_gap_ptr + base_in + tl.arange(0, 8))
    
    # Dot product
    acc += tl.sum(w * xg)
    
    # Sigmoid activation
    attn = tl.sigmoid(acc)
    
    # Apply attention to all spatial positions in this block
    x_base = c_idx * HW + hw_offsets
    x_vals = tl.load(x_ptr + x_base, mask=mask)
    results = x_vals * attn
    
    tl.store(out_ptr + x_base, results, mask=mask)


@torch.fx.wrap
def fused_se(bias, weight, x, x_gap):
    # Shapes: x is [1, 96, 128, 128], x_gap is [1, 32, 1, 1]
    batch, C, H, W = x.shape
    HW = H * W
    
    out = torch.empty_like(x)
    
    # HW = 128*128 = 16384
    BLOCK_HW = 2048
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    
    grid = (C, num_hw_blocks)
    
    fused_se_kernel[grid](
        bias, weight, x, x_gap.view(-1), out,
        HW,
        BLOCK_HW,
        num_warps=8,
    )
    
    return out

def replacement_func():
    return fused_se