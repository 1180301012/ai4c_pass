import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror model.py exactly
def pattern(in_0, in_1, in_2):
    """
    Match the avg_pool2d + scaled residual pattern
    """
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    tmp_3 = tmp_2 - in_2
    tmp_4 = in_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 * tmp_3
    tmp_7 = in_2 + tmp_6
    tmp_8 = in_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# 1D kernel with autotune for best performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_avgpool_1d_kernel(
    in_ptr,        # Input tensor [B, C, H, W]
    scale_ptr,     # Scale tensor [C]
    out_ptr,       # Output tensor [B, C, H, W]
    total_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    CHW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D kernel that computes:
    output = input + scale * (avg_pool2d(input) - input)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute b, c, h, w indices from flat offset
    # offset = b * CHW + c * HW + h * W + w
    b = offsets // CHW
    remainder = offsets % CHW
    c = remainder // HW
    remainder2 = remainder % HW
    h = remainder2 // W
    w = remainder2 % W
    
    # Load scale for each channel
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    
    # Load center value
    center_val = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute avg_pool2d with 3x3 kernel, count_include_pad=False
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over 3x3 neighborhood
    for dh in tl.static_range(-1, 2):
        for dw in tl.static_range(-1, 2):
            nh = h + dh
            nw = w + dw
            
            # Check bounds
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W) & mask
            
            # Compute neighbor offset
            neighbor_offset = b * CHW + c * HW + nh * W + nw
            
            # Load neighbor value
            neighbor_val = tl.load(in_ptr + neighbor_offset, mask=valid, other=0.0)
            
            pool_sum = pool_sum + neighbor_val
            count = count + tl.where(valid, 1.0, 0.0)
    
    # Compute average
    pool_avg = pool_sum / count
    
    # Compute output = input + scale * (pool_avg - input)
    diff = pool_avg - center_val
    output = center_val + scale * diff
    
    # Store result
    tl.store(out_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_avgpool_main(in_0, in_2):
    """
    Fused implementation of avg_pool2d + scaled residual
    """
    B, C, H, W = in_2.shape
    
    # Allocate output tensor
    out_7 = torch.empty_like(in_2)
    
    total_elements = B * C * H * W
    HW = H * W
    CHW = C * H * W
    
    # Fixed grid for 1D kernel - autotune selects best BLOCK_SIZE
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_avgpool_1d_kernel[grid](
        in_2,
        in_0,
        out_7,
        total_elements,
        C, H, W, HW, CHW,
    )
    
    return out_7


def fused_avgpool_scaled_residual(in_0, in_1, in_2):
    """
    Replacement function that returns two separate outputs.
    """
    out_7 = fused_avgpool_main(in_0, in_2)
    tmp_8 = in_1.unsqueeze(-1)
    out_9 = tmp_8.unsqueeze(-1)
    return out_7, out_9


def replacement_func():
    return fused_avgpool_scaled_residual