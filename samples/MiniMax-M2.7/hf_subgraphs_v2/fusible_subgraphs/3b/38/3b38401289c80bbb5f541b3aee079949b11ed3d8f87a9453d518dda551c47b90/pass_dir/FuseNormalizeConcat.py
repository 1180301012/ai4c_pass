import torch
import triton
import triton.language as tl

# Pattern matching - matches the full computation graph
def pattern(in_0, in_1):
    """
    Match the normalization and concatenation pattern:
    1. in_1 * scale1 + bias1
    2. in_0[:,1,:,:] -> unsqueeze -> * scale2 + bias2
    3. in_0[:,2,:,:] -> unsqueeze -> * scale3 + bias3
    4. cat along dim=1
    """
    # Path 1: in_1 processing
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Path 2: channel 1 from in_0
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Path 3: channel 2 from in_0
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Concatenate along dim=1
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11

# Extract arguments from matched nodes
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# 2D tile-based kernel for better memory access patterns
@triton.jit
def fuse_normalize_concat_kernel_2d(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_stride, h_stride, w_stride,
    out_batch_stride, out_h_stride, out_w_stride,
    B, H, W, bh_per_batch,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    2D tile-based kernel with explicit bounds checking.
    Grid: (B * bh_per_batch) x (ceil(W/BLOCK_W))
    """
    # Get tile coordinates
    pid = tl.program_id(0)
    w_pid = tl.program_id(1)
    
    # Calculate batch and local tile indices
    batch_idx = pid // bh_per_batch
    tile_h_idx = pid % bh_per_batch
    h_start = tile_h_idx * BLOCK_H
    w_start = w_pid * BLOCK_W
    
    # Create offset ranges
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Create masks for bounds checking
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Calculate actual h indices with bounds
    h_idx = h_offsets
    w_idx_expanded = w_offsets[None, :]
    
    # Compute base offset for each position in in_0
    base_offset = batch_idx * batch_stride + h_idx[:, None] * h_stride + w_idx_expanded * w_stride
    
    # Load data from in_0
    in_0_ch1 = tl.load(in_0_ptr + base_offset + 1 * H * W, mask=mask, other=0.0)
    in_0_ch2 = tl.load(in_0_ptr + base_offset + 2 * H * W, mask=mask, other=0.0)
    
    # Load from in_1 (stride: H*W, H, 1)
    in_1_offset = batch_idx * H * W + h_idx[:, None] * H + w_idx_expanded
    in_1_val = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    
    # Compute outputs
    out_ch0 = in_1_val * 0.458 - 0.030000000000000027
    out_ch1 = in_0_ch1 * 0.448 - 0.08799999999999997
    out_ch2 = in_0_ch2 * 0.45 - 0.18799999999999994
    
    # Calculate output offsets
    out_base = batch_idx * out_batch_stride + h_idx[:, None] * out_h_stride + w_idx_expanded * out_w_stride
    
    # Store results
    tl.store(out_ptr + out_base, out_ch0, mask=mask)
    tl.store(out_ptr + out_base + H * W, out_ch1, mask=mask)
    tl.store(out_ptr + out_base + 2 * H * W, out_ch2, mask=mask)


def fuse_normalize_concat_impl(in_0, in_1):
    """
    Dispatch the fused normalization and concatenation kernel.
    Output shape: [B, 3, H, W]
    """
    B, C, H, W = in_0.shape
    
    # Ensure contiguous tensors for predictable strides
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    # Output tensor
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)
    out = out.contiguous()
    
    # Strides
    batch_stride = in_0.stride(0)   # C * H * W
    h_stride = in_0.stride(2)       # W
    w_stride = in_0.stride(3)       # 1
    
    out_batch_stride = out.stride(0)
    out_h_stride = out.stride(2)
    out_w_stride = out.stride(3)
    
    # Choose tile sizes based on H dimension
    if H >= 64:
        BLOCK_H, BLOCK_W = 16, 16
    elif H >= 32:
        BLOCK_H, BLOCK_W = 8, 16
    else:
        BLOCK_H, BLOCK_W = 4, 16
    
    # Grid dimensions
    bh_per_batch = (H + BLOCK_H - 1) // BLOCK_H
    num_bh = bh_per_batch * B
    num_w = (W + BLOCK_W - 1) // BLOCK_W
    
    # Launch kernel
    fuse_normalize_concat_kernel_2d[(num_bh, num_w)](
        in_0, in_1, out,
        batch_stride, h_stride, w_stride,
        out_batch_stride, out_h_stride, out_w_stride,
        B, H, W, bh_per_batch,
        BLOCK_H, BLOCK_W,
    )
    
    return out


@torch.fx.wrap
def fuse_normalize_concat_dispatcher(in_0, in_1):
    return fuse_normalize_concat_impl(in_0, in_1)


def replacement_func():
    return fuse_normalize_concat_dispatcher