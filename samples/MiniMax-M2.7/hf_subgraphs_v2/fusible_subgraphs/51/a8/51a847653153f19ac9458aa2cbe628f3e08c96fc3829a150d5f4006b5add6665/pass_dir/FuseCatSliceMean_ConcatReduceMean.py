import torch
import triton
import triton.language as tl

# Optimized kernel that performs concat + slice + mean in a single pass
# We use two separate kernels: one for the sliced concat, one for the mean

# Kernel 1: Produce the sliced concat result (concat in_0 and in_1 along channel dim)
@triton.jit
def concat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program copies one channel from in_0 or in_1 to out."""
    pid = tl.program_id(0)
    
    batch_idx = pid // (2 * C)
    channel_idx = pid % (2 * C)
    
    # Determine source and destination
    if channel_idx < C:
        src_ptr = in_0_ptr
        src_ch = channel_idx
    else:
        src_ptr = in_1_ptr
        src_ch = channel_idx - C
    
    # Compute offsets
    src_base = batch_idx * C * H * W + src_ch * H * W
    dst_base = batch_idx * 2 * C * H * W + channel_idx * H * W
    
    # Copy all spatial data
    for h in range(H):
        src_offset = src_base + h * W
        dst_offset = dst_base + h * W
        for w in range(W):
            val = tl.load(src_ptr + src_offset + w)
            tl.store(out_ptr + dst_offset + w, val)


# Kernel 2: Compute mean over spatial dimensions
@triton.jit
def mean_reduce_kernel(
    in_ptr,
    out_ptr,
    B,
    C_total,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program computes mean for one (batch, channel)."""
    pid = tl.program_id(0)
    
    batch_idx = pid // C_total
    channel_idx = pid % C_total
    
    base_offset = batch_idx * C_total * H * W + channel_idx * H * W
    
    # Accumulate sum over spatial dimensions
    sum_val = 0.0
    for h in range(H):
        row_offset = base_offset + h * W
        for w in range(W):
            val = tl.load(in_ptr + row_offset + w)
            sum_val += val
    
    mean_val = sum_val / (H * W)
    
    # Store at (batch, channel, 0, 0) position, flattened
    out_offset = batch_idx * C_total + channel_idx
    tl.store(out_ptr + out_offset, mean_val)


@torch.fx.wrap
def fused_cat_slice_mean_triton(in_0, in_1):
    """Fused implementation: concat + slice + mean without intermediate tensors."""
    B, C, H, W = in_0.shape
    C_total = 2 * C
    
    # Allocate output tensors
    # out_slice: [B, 2*C, H, W] - concatenated result
    # out_mean: [B, 2*C, 1, 1] - mean over spatial dims
    out_slice = torch.empty((B, C_total, H, W), device=in_0.device, dtype=in_0.dtype)
    out_mean = torch.empty((B, C_total, 1, 1), device=in_0.device, dtype=in_0.dtype)
    
    # Grid for concat: B * 2 * C programs (one per output channel per batch)
    concat_grid = (B * C_total,)
    concat_kernel[concat_grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out_slice,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    
    # Grid for mean: B * 2 * C programs
    mean_grid = (B * C_total,)
    mean_reduce_kernel[mean_grid](
        in_ptr=out_slice,
        out_ptr=out_mean,
        B=B,
        C_total=C_total,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    
    return out_slice, out_mean


def pattern(in_0, in_1):
    """
    Match pattern: concat + getitem slice + mean(dim=[2,3], keepdim=True)
    The slice extracts the first 2*C channels (i.e., the full concatenated tensor).
    We optimize by computing the mean directly without materializing intermediate tensors.
    """
    # Step 1: Concatenate along channel dimension
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Step 2: Slice to extract first 2*C channels (full concat)
    tmp_1 = tmp_0[:, :in_0.size(1) * 2, :, :]
    # Step 3: Mean over spatial dimensions with keepdim
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_cat_slice_mean_triton