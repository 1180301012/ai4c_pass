import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern matching softmax -> mul -> sum along dim=1
    in_0: [B, 2, C, H, W]
    in_1: [B, 2, C, 1, 1]
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_kernel_2d(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    C: tl.constexpr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    2D kernel - one program per (channel, hw_block)
    Grid: (C, ceil(HW/BLOCK_HW))
    """
    c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    # Load weights and compute sigmoid
    a = tl.load(in_1_ptr + c)
    b = tl.load(in_1_ptr + C + c)
    w = tl.sigmoid(a - b)
    
    # Spatial offsets
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW
    
    # Memory offsets
    CHW = C * HW
    base = c * HW + hw_offs
    
    # Load data
    x0 = tl.load(in_0_ptr + base, mask=hw_mask, other=0.0)
    x1 = tl.load(in_0_ptr + CHW + base, mask=hw_mask, other=0.0)
    
    # Compute weighted sum
    out_val = x1 + (x0 - x1) * w
    tl.store(out_ptr + base, out_val, mask=hw_mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    """
    Fused implementation using Triton kernel
    """
    B, K, C, H, W = in_0.shape
    HW = H * W
    
    out = torch.empty((B, C, H, W), device=in_0.device, dtype=in_0.dtype)
    
    BLOCK_HW = 1024
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (C, num_hw_blocks)
    
    fused_kernel_2d[grid](
        in_0, in_1, out,
        C=C, HW=HW, BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    
    return out


def replacement_func():
    return fused_softmax_mul_sum