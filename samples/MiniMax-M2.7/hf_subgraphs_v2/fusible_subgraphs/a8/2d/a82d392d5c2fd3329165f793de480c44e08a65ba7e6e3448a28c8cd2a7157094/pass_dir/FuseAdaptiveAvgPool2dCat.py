import torch
import triton
import triton.language as tl


@triton.jit
def fused_adaptive_avg_pool2d_cat_kernel(
    # Input tensor in_0: [B, C_in=20, H_in=64, W_in=48]
    in_0_ptr,
    # Input tensor in_1: [B, C_in2=40, H=32, W=24]
    in_1_ptr,
    # Output tensor: [B, C_out=60, H=32, W=24]
    out_ptr,
    # Tensor dimensions
    B: tl.constexpr,
    C0: tl.constexpr,  # 20
    C1: tl.constexpr,  # 40
    H0: tl.constexpr,  # 64 (input height)
    W0: tl.constexpr,  # 48 (input width)
    H_out: tl.constexpr,  # 32
    W_out: tl.constexpr,  # 24
    stride_in0_b: tl.constexpr,
    stride_in0_c: tl.constexpr,
    stride_in0_h: tl.constexpr,
    stride_in0_w: tl.constexpr,
    stride_in1_b: tl.constexpr,
    stride_in1_c: tl.constexpr,
    stride_in1_h: tl.constexpr,
    stride_in1_w: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_c: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_w: tl.constexpr,
    # Total output elements for grid
    TOTAL_OUTPUT_SIZE: tl.constexpr,
    # Block size for parallelization
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs adaptive_avg_pool2d on in_0 and concatenates
    the result with in_1 along the channel dimension.
    
    Uses 1D grid: (total_elements // BLOCK_SIZE,)
    Each program processes BLOCK_SIZE elements using vectorized operations.
    """
    # Calculate pooled window size
    pool_h = H0 // H_out  # 64 // 32 = 2
    pool_w = W0 // W_out  # 48 // 24 = 2
    
    # Block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for out-of-bounds
    mask = offsets < TOTAL_OUTPUT_SIZE
    
    # Compute 4D indices from linear offset
    # Output is [B, C0+C1, H_out, W_out]
    C_total = C0 + C1
    CHW = C_total * H_out * W_out
    HW = H_out * W_out
    
    # Compute indices: each thread computes its own
    batch_idx = offsets // CHW
    local_idx = offsets % CHW
    
    c_idx = local_idx // HW
    hh_idx = (local_idx // W_out) % H_out
    w_idx = local_idx % W_out
    
    # Compute output offsets for all threads
    out_offsets = (
        batch_idx * stride_out_b +
        c_idx * stride_out_c +
        hh_idx * stride_out_h +
        w_idx * stride_out_w
    )
    
    # Create mask for which channels are from in_0 vs in_1
    in0_channel_mask = c_idx < C0
    
    # For pooled in_0 channels, compute sum over pooling window
    # We process each element sequentially in the block
    sum_0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Only load from in_0 when c_idx < C0 (otherwise we're in in_1 territory)
    in0_mask = in0_channel_mask & mask
    
    # Load from in_0 and accumulate - each thread loads pool_h * pool_w values
    for ph in range(pool_h):
        for pw in range(pool_w):
            # Compute in_0 offset: b*stride_b + c*stride_c + (h*pool_h+ph)*stride_h + (w*pool_w+pw)*stride_w
            in0_offsets = (
                batch_idx * stride_in0_b +
                c_idx * stride_in0_c +
                (hh_idx * pool_h + ph) * stride_in0_h +
                (w_idx * pool_w + pw) * stride_in0_w
            )
            val = tl.load(in_0_ptr + in0_offsets, mask=in0_mask, other=0.0)
            sum_0 = sum_0 + val
    
    # Compute pooled average for in_0 channels
    out_0 = sum_0 / (pool_h * pool_w)
    
    # Compute in_1 offsets - use tl.where to ensure valid offset when c_idx < C0
    # When c_idx < C0, we use channel 0 (will be masked anyway)
    in1_c_idx = tl.where(c_idx >= C0, c_idx - C0, 0)
    in1_offsets = (
        batch_idx * stride_in1_b +
        in1_c_idx * stride_in1_c +
        hh_idx * stride_in1_h +
        w_idx * stride_in1_w
    )
    
    # Load from in_1 (only for channels where c >= C0)
    in1_mask = (c_idx >= C0) & mask
    out_1 = tl.load(in_1_ptr + in1_offsets, mask=in1_mask, other=0.0)
    
    # Select output based on channel
    out_vals = tl.where(in0_channel_mask, out_0, out_1)
    
    # Store final output
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_adaptive_avg_pool2d_cat(in_0, in_1):
    """
    Fused implementation of:
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
        tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    
    Input shapes:
        in_0: [B, 20, 64, 48]
        in_1: [B, 40, 32, 24]
    
    Output shape: [B, 60, 32, 24]
    """
    B, C0, H0, W0 = in_0.shape
    _, C1, H_out, W_out = in_1.shape
    
    C_total = C0 + C1
    total_output_size = B * C_total * H_out * W_out
    
    # Allocate output tensor
    out = torch.empty((B, C_total, H_out, W_out), device=in_0.device, dtype=in_0.dtype)
    
    # Launch configuration
    BLOCK_SIZE = 1024
    num_programs = (total_output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_adaptive_avg_pool2d_cat_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        B,
        C0,
        C1,
        H0,
        W0,
        H_out,
        W_out,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        total_output_size,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: adaptive_avg_pool2d followed by cat along dim=1
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_adaptive_avg_pool2d_cat