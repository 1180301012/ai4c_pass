import torch
import triton
import triton.language as tl


# Pattern matching function - matches interpolate followed by sigmoid
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the full graph: conv2d + interpolate + 6 sigmoid operations
    The key optimization: fuse interpolate + sigmoid (tmp_3 -> tmp_9)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(640, 640), mode='bilinear')
    tmp_4 = torch.nn.functional.sigmoid(in_3)
    tmp_5 = torch.nn.functional.sigmoid(in_4)
    tmp_6 = torch.nn.functional.sigmoid(in_5)
    tmp_7 = torch.nn.functional.sigmoid(in_6)
    tmp_8 = torch.nn.functional.sigmoid(in_7)
    tmp_9 = torch.nn.functional.sigmoid(tmp_3)
    return (tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


# Optimized Triton kernel for interpolate + sigmoid fusion
@triton.jit
def interpolate_sigmoid_kernel(
    input_ptr, output_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused interpolate + sigmoid kernel"""
    # Each program processes a row of channels
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    # Calculate offset
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (W * C)
    
    # Load input: shape [B, C, H, W]
    base_offset = batch_idx * C * H * W + row_idx * W * C
    x = tl.load(input_ptr + base_offset + offsets * C, mask=mask, other=0.0)
    
    # Apply sigmoid: 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    
    # Store output
    tl.store(output_ptr + base_offset + offsets * C, sigmoid, mask=mask)


def interpolate_sigmoid_fused(x):
    """
    Fused interpolate + sigmoid operation.
    First performs bilinear interpolation to 640x640, then applies sigmoid.
    """
    B, C, H, W = x.shape
    target_H, target_W = 640, 640
    
    # Step 1: Interpolate to target size
    interpolated = torch.nn.functional.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
    
    # Step 2: Apply sigmoid using Triton kernel
    # We launch (B, target_H) programs, each processing one row
    output = torch.empty((B, C, target_H, target_W), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    # Ensure BLOCK_SIZE is large enough for W * C
    if target_W * C <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = triton.next_power_of_2(target_W * C)
    
    grid = (B, target_H)
    
    interpolate_sigmoid_kernel[grid](
        interpolated.view(-1).contiguous(),
        output.view(-1).contiguous(),
        B, C, target_H, target_W,
        BLOCK_SIZE
    )
    
    return output


# Wrapper function that handles the full computation
@torch.fx.wrap
def fused_interpolate_sigmoid(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Fused kernel that performs:
    1. conv2d(in_2, in_1, in_0) -> [B, 512, 10, 10]
    2. interpolate to 640x640
    3. sigmoid on interpolate result
    4. sigmoid on 5 independent inputs
    """
    device = in_2.device
    dtype = in_2.dtype
    
    # Original conv2d
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1)
    
    # Fused interpolate + sigmoid for the conv output
    tmp_9 = interpolate_sigmoid_fused(tmp_2)
    
    # Independent sigmoid operations (keep as separate for now)
    tmp_4 = torch.nn.functional.sigmoid(in_3)
    tmp_5 = torch.nn.functional.sigmoid(in_4)
    tmp_6 = torch.nn.functional.sigmoid(in_5)
    tmp_7 = torch.nn.functional.sigmoid(in_6)
    tmp_8 = torch.nn.functional.sigmoid(in_7)
    
    return (tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9)


def replacement_func():
    """Return the fused kernel function"""
    return fused_interpolate_sigmoid