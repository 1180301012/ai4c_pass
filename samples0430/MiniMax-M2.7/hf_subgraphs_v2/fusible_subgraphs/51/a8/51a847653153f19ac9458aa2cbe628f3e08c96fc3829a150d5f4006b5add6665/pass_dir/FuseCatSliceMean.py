import torch
import triton
import triton.language as tl


@triton.jit
def fuse_cat_slice_mean_kernel(
    in_0_ptr,
    in_1_ptr,
    out_slice_ptr,
    out_mean_ptr,
    N_elements,  # Number of channels to slice (equals in_0's channels)
    H,
    W,
    stride_in_0_c,
    stride_in_0_h,
    stride_in_0_w,
    stride_in_1_c,
    stride_in_1_h,
    stride_in_1_w,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Fused kernel for:
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[:, :N, :, :]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    
    Returns (tmp_1, tmp_2)
    """
    # Get batch and channel indices
    batch_idx = tl.program_id(0)
    channel_block_idx = tl.program_id(1)
    
    # Calculate channel offsets for this block
    c_offset = channel_block_idx * BLOCK_SIZE_C
    c_mask = c_offset + tl.arange(0, BLOCK_SIZE_C) < N_elements
    
    # Create channel offsets for loading
    c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
    
    # Create spatial offsets
    hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W
    
    # Compute strides
    in_0_offsets = (
        batch_idx * stride_in_0_c * N_elements +
        c_offsets[:, None] * stride_in_0_c +
        h_offsets[None, :] * stride_in_0_h +
        w_offsets[None, :] * stride_in_0_w
    )
    
    # Load from in_0 (we only need first N channels, which is in_0)
    # in_0 has shape [B, N, H, W]
    mask = (
        (c_offsets[:, None] < N_elements) &
        (h_offsets[None, :] < H) &
        (w_offsets[None, :] < W)
    )
    
    in_0_vals = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=0.0)
    
    # Store sliced output (tmp_1)
    out_slice_offsets = (
        batch_idx * stride_out_c * N_elements +
        c_offsets[:, None] * stride_out_c +
        h_offsets[None, :] * stride_out_h +
        w_offsets[None, :] * stride_out_w
    )
    tl.store(out_slice_ptr + out_slice_offsets, in_0_vals, mask=mask)
    
    # Compute mean over spatial dimensions for output_mean (tmp_2)
    # mean over (H, W) with keepdim=True -> output shape [B, N, 1, 1]
    # We need to reduce across H*W
    sum_vals = tl.sum(in_0_vals, axis=1)  # Sum over spatial dim (H*W flattened)
    
    # Store mean output at position [b, c, 0, 0]
    mean_out_ptr = (
        batch_idx * N_elements * 1 * 1 +
        c_offsets * 1 * 1
    )
    tl.store(out_mean_ptr + mean_out_ptr, sum_vals / (H * W), mask=c_mask)


@torch.fx.wrap
def fused_cat_slice_mean(in_0, in_1):
    """
    Fused implementation of:
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[:, :N, :, :]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    
    Returns (tmp_1, tmp_2)
    """
    B, C0, H, W = in_0.shape
    _, C1, _, _ = in_1.shape
    
    # N is the number of channels we slice (always equal to in_0's channels based on pattern)
    N = C0
    
    # Allocate output tensors
    out_slice = torch.empty((B, N, H, W), dtype=in_0.dtype, device=in_0.device)
    out_mean = torch.empty((B, N, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_HW = 64  # H * W block size
    
    grid = (B, (N + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C)
    
    fuse_cat_slice_mean_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_slice_ptr=out_slice,
        out_mean_ptr=out_mean,
        N_elements=N,
        H=H,
        W=W,
        stride_in_0_c=in_0.stride(1),
        stride_in_0_h=in_0.stride(2),
        stride_in_0_w=in_0.stride(3),
        stride_in_1_c=in_1.stride(1),
        stride_in_1_h=in_1.stride(2),
        stride_in_1_w=in_1.stride(3),
        stride_out_c=out_slice.stride(1),
        stride_out_h=out_slice.stride(2),
        stride_out_w=out_slice.stride(3),
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out_slice, out_mean


def pattern(in_0, in_1):
    """
    Pattern to match:
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[:, :N, :, :]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)
    
    The slice always takes the first N channels (where N = in_0.shape[1])
    """
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    N = in_0.shape[1]
    tmp_1 = tmp_0[:, :N, :, :]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_cat_slice_mean