import torch
import triton
import triton.language as tl


# Pattern matching function - matches SiLU followed by split
def pattern(in_0, in_1):
    silu_out = torch.nn.functional.silu(in_1)
    split_out = torch.functional.split(silu_out, [512, 512, 128], dim=2)
    part_0 = split_out[0]
    part_1 = split_out[1]
    part_2 = split_out[2]
    unsqueezed = part_2.unsqueeze(2)
    broadcasted = in_0[None, None, :]
    return broadcasted, part_0, unsqueezed, part_1


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# SiLU activation kernel fused with split
@triton.jit
def silu_split_kernel(
    in_ptr, out0_ptr, out1_ptr, out2_ptr, out3_ptr,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused SiLU + split kernel.
    in_1 shape: [B, H, C] where C = 1152
    Split sizes: [512, 512, 128] along C dimension
    
    Outputs:
    - out0: [1, 1, B, H] for tmp_7 (actually from in_0)
    - out1: [B, H, 512] for tmp_3
    - out2: [B, H, 1, 128] for tmp_6
    - out3: [B, H, 512] for tmp_4
    """
    # Each program processes a (B, H) position
    program_id = tl.program_id(0)
    b = program_id // H
    h = program_id % H
    
    if b >= B:
        return
    
    # Calculate base offset for this (b, h)
    base_offset = b * H * C + h * C
    
    # Process each segment with separate BLOCK_SIZE
    # First segment: 512 elements
    offsets_0 = base_offset + tl.arange(0, BLOCK_SIZE)
    mask_0 = offsets_0 < base_offset + 512
    x_0 = tl.load(in_ptr + offsets_0, mask=mask_0, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu_0 = x_0 * tl.sigmoid(x_0)
    out_offsets_0 = b * H * 512 + h * 512 + tl.arange(0, BLOCK_SIZE)
    out_mask_0 = tl.arange(0, BLOCK_SIZE) < 512
    tl.store(out1_ptr + out_offsets_0, silu_0, mask=out_mask_0)
    
    # Second segment: next 512 elements (offset 512)
    offsets_1 = base_offset + 512 + tl.arange(0, BLOCK_SIZE)
    mask_1 = offsets_1 < base_offset + 1024
    x_1 = tl.load(in_ptr + offsets_1, mask=mask_1, other=0.0)
    silu_1 = x_1 * tl.sigmoid(x_1)
    out_offsets_1 = b * H * 512 + h * 512 + tl.arange(0, BLOCK_SIZE)
    out_mask_1 = tl.arange(0, BLOCK_SIZE) < 512
    tl.store(out3_ptr + out_offsets_1, silu_1, mask=out_mask_1)
    
    # Third segment: last 128 elements (offset 1024)
    offsets_2 = base_offset + 1024 + tl.arange(0, BLOCK_SIZE)
    mask_2 = offsets_2 < base_offset + 1152
    x_2 = tl.load(in_ptr + offsets_2, mask=mask_2, other=0.0)
    silu_2 = x_2 * tl.sigmoid(x_2)
    # For unsqueeze(2), we need to store as [B, H, 1, 128]
    out_offsets_2 = b * H * 128 + h * 128 + tl.arange(0, BLOCK_SIZE)
    out_mask_2 = tl.arange(0, BLOCK_SIZE) < 128
    tl.store(out2_ptr + out_offsets_2, silu_2, mask=out_mask_2)


@torch.fx.wrap
def silu_split_wrapper(in_0, in_1):
    """
    Wrapper function for the fused SiLU + split kernel.
    
    Args:
        in_0: weight tensor, shape [2, 128]
        in_1: input tensor, shape [B, 17, 1152]
    
    Returns:
        tuple of (tmp_7, tmp_3, tmp_6, tmp_4)
        - tmp_7: [1, 1, 2, 128]
        - tmp_3: [B, 17, 512]
        - tmp_6: [B, 17, 1, 128]
        - tmp_4: [B, 17, 512]
    """
    B, H, C = in_1.shape
    assert C == 1152, f"Expected C=1152, got {C}"
    
    # Allocate output tensors
    # tmp_3: [B, 17, 512]
    out1 = torch.empty((B, H, 512), dtype=in_1.dtype, device=in_1.device)
    # tmp_4: [B, 17, 512]
    out3 = torch.empty((B, H, 512), dtype=in_1.dtype, device=in_1.device)
    # tmp_6: [B, 17, 128] (will be unsqueezed to [B, 17, 1, 128])
    out2 = torch.empty((B, H, 128), dtype=in_1.dtype, device=in_1.device)
    
    # tmp_7: [1, 1, 2, 128] - from in_0 with dimensions added
    out0 = in_0[None, None, :]
    
    # Calculate grid
    # Each (b, h) position needs one program
    grid = (B * H,)
    
    BLOCK_SIZE = 128
    
    silu_split_kernel[grid](
        in_1,
        out0, out1, out2, out3,
        B, H, C,
        BLOCK_SIZE,
    )
    
    # Apply unsqueeze(2) to tmp_6
    tmp_6 = out2.unsqueeze(2)
    
    return (out0, out1, tmp_6, out3)


def replacement_func():
    return silu_split_wrapper