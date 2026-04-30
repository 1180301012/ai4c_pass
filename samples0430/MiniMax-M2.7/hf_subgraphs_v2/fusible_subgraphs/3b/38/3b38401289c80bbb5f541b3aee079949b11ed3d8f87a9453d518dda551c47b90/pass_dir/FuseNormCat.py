import torch
import triton
import triton.language as tl


@triton.jit
def fused_norm_cat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    scale1: tl.constexpr,
    bias1: tl.constexpr,
    scale2: tl.constexpr,
    bias2: tl.constexpr,
    scale3: tl.constexpr,
    bias3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Block index: each block processes a contiguous HW slice
    batch_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)
    
    # Compute h, w coordinates
    h = offsets // W
    w = offsets % W
    
    # Load in_1 (shape: [B, 1, H, W])
    in_1_idx = batch_id * H * W + h * W + w
    in_1_val = tl.load(in_1_ptr + in_1_idx, mask=mask, other=0.0)
    
    # Normalize in_1: val * scale1 + bias1
    norm1 = in_1_val * scale1 + bias1
    
    # Load channel 1 from in_0 (shape: [B, C, H, W])
    in_0_ch1_idx = batch_id * C * H * W + 1 * H * W + h * W + w
    ch1_val = tl.load(in_0_ptr + in_0_ch1_idx, mask=mask, other=0.0)
    
    # Normalize channel 1: val * scale2 + bias2
    norm2 = ch1_val * scale2 + bias2
    
    # Load channel 2 from in_0
    in_0_ch2_idx = batch_id * C * H * W + 2 * H * W + h * W + w
    ch2_val = tl.load(in_0_ptr + in_0_ch2_idx, mask=mask, other=0.0)
    
    # Normalize channel 2: val * scale3 + bias3
    norm3 = ch2_val * scale3 + bias3
    
    # Store concatenated result to out_ptr (shape: [B, 3, H, W])
    out_idx0 = batch_id * 3 * H * W + 0 * H * W + h * W + w
    out_idx1 = batch_id * 3 * H * W + 1 * H * W + h * W + w
    out_idx2 = batch_id * 3 * H * W + 2 * H * W + h * W + w
    
    tl.store(out_ptr + out_idx0, norm1, mask=mask)
    tl.store(out_ptr + out_idx1, norm2, mask=mask)
    tl.store(out_ptr + out_idx2, norm3, mask=mask)


@torch.fx.wrap
def fused_norm_cat(in_0, in_1):
    B, C, H, W = in_0.shape
    # in_0 needs at least 3 channels (we extract channels 1 and 2)
    B_in1, C_in1, H_in1, W_in1 = in_1.shape
    assert C >= 3, f"in_0 must have at least 3 channels, got {C}"
    assert C_in1 == 1, f"in_1 must have 1 channel, got {C_in1}"
    assert H == H_in1 and W == W_in1, f"Spatial dimensions mismatch"
    
    # Normalization constants
    scale1 = 0.458
    bias1 = -0.030000000000000027
    scale2 = 0.448
    bias2 = -0.08799999999999997
    scale3 = 0.45
    bias3 = -0.18799999999999994
    
    # Output shape: [B, 3, H, W]
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Choose BLOCK_SIZE based on problem size
    HW = H * W
    if HW <= 256:
        BLOCK_SIZE = 256
    elif HW <= 512:
        BLOCK_SIZE = 512
    elif HW <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_blocks = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid: (B, num_blocks)
    grid = (B, num_blocks)
    
    fused_norm_cat_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        scale1=scale1,
        bias1=bias1,
        scale2=scale2,
        bias2=bias2,
        scale3=scale3,
        bias3=bias3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_norm_cat