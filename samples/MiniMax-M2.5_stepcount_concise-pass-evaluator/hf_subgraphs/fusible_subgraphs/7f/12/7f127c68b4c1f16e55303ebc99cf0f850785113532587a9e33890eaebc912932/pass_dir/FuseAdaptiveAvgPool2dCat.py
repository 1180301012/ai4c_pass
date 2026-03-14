import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: adaptive_avg_pool2d + cat
    This pattern:
    1. Applies adaptive average pooling on in_0 from (H, W) to (32, 24)
    2. Concatenates the result with in_1 along dim=1
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Fully fused Triton kernel: adaptive_avg_pool2d + cat
# Uses block-based parallelization with efficient reduction
@triton.autotune(
    configs=[
        # (BLOCK_M, BLOCK_N, num_warps)
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 16}, num_warps=4),
    ],
    key=['batch_size', 'out_h', 'out_w'],
)
@triton.jit
def fused_adaptive_avg_pool2d_cat_kernel(
    # Input pointers
    in_0_ptr, in_1_ptr, out_ptr,
    # Tensor dimensions
    batch_size, in_channels, in_h, in_w,
    out_channels, out_h, out_w,
    pooled_channels,
    # Pooling parameters (compile-time)
    pool_stride_h: tl.constexpr, pool_stride_w: tl.constexpr,
    pool_size_h: tl.constexpr, pool_size_w: tl.constexpr,
    # Strides
    in_0_stride_b, in_0_stride_c, in_0_stride_h, in_0_stride_w,
    in_1_stride_b, in_1_stride_c, in_1_stride_h, in_1_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Block dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Fully fused kernel that performs adaptive_avg_pool2d and cat.
    
    Grid: (batch_size * out_h * out_w,)
    Each program handles BLOCK_M * BLOCK_N output elements.
    """
    # Compute program ID
    pid = tl.program_id(0)
    spatial_size = out_h * out_w
    
    # Compute batch and spatial indices
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    out_h_idx = spatial_idx // out_w
    out_w_idx = spatial_idx % out_w
    
    # Bounds check
    if batch_idx >= batch_size:
        return
    
    # Calculate input pooling start positions
    in_h_start = out_h_idx * pool_stride_h
    in_w_start = out_w_idx * pool_stride_w
    
    # Process output channels in blocks
    # BLOCK_N channels per iteration
    for ch_base in range(0, out_channels, BLOCK_N):
        # Calculate which channels to process
        ch_limit = tl.minimum(ch_base + BLOCK_N, out_channels)
        
        # Process each channel in the block
        for ch_idx in range(ch_base, ch_limit):
            # Determine if this channel comes from pooled input or direct copy
            if ch_idx < pooled_channels:
                # Pooled channel: compute 2x2 average
                sum_val = 0.0
                # Load all 4 values and accumulate
                for dh in range(pool_size_h):
                    for dw in range(pool_size_w):
                        offset = (
                            batch_idx * in_0_stride_b +
                            ch_idx * in_0_stride_c +
                            (in_h_start + dh) * in_0_stride_h +
                            (in_w_start + dw) * in_0_stride_w
                        )
                        val = tl.load(in_0_ptr + offset)
                        sum_val = sum_val + val
                avg_val = sum_val / (pool_size_h * pool_size_w)
                
                # Store to output
                out_offset = (
                    batch_idx * out_stride_b +
                    ch_idx * out_stride_c +
                    out_h_idx * out_stride_h +
                    out_w_idx * out_stride_w
                )
                tl.store(out_ptr + out_offset, avg_val)
            else:
                # Direct copy from in_1
                in1_ch = ch_idx - pooled_channels
                offset = (
                    batch_idx * in_1_stride_b +
                    in1_ch * in_1_stride_c +
                    out_h_idx * in_1_stride_h +
                    out_w_idx * in_1_stride_w
                )
                val = tl.load(in_1_ptr + offset)
                
                # Store to output
                out_offset = (
                    batch_idx * out_stride_b +
                    ch_idx * out_stride_c +
                    out_h_idx * out_stride_h +
                    out_w_idx * out_stride_w
                )
                tl.store(out_ptr + out_offset, val)


@torch.fx.wrap
def fused_adaptive_avg_pool2d_cat_kernel_wrapper(in_0, in_1):
    """
    Wrapper that launches the fully fused Triton kernel.
    """
    batch_size, in_c, in_h, in_w = in_0.shape
    _, in1_c, in1_h, in1_w = in_1.shape
    
    out_h, out_w = in1_h, in1_w
    pooled_c = in_c
    out_c = pooled_c + in1_c
    
    # Compute pooling stride (for output 32x24 on input 64x48, stride = 2)
    pool_stride_h = in_h // out_h
    pool_stride_w = in_w // out_w
    
    out = torch.empty((batch_size, out_c, out_h, out_w), 
                      dtype=in_0.dtype, device=in_0.device)
    
    # Grid: one program per spatial location per batch
    num_programs = batch_size * out_h * out_w
    
    fused_adaptive_avg_pool2d_cat_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_c,
        in_h=in_h,
        in_w=in_w,
        out_channels=out_c,
        out_h=out_h,
        out_w=out_w,
        pooled_channels=pooled_c,
        pool_stride_h=pool_stride_h,
        pool_stride_w=pool_stride_w,
        pool_size_h=pool_stride_h,
        pool_size_w=pool_stride_w,
        in_0_stride_b=in_0.stride(0),
        in_0_stride_c=in_0.stride(1),
        in_0_stride_h=in_0.stride(2),
        in_0_stride_w=in_0.stride(3),
        in_1_stride_b=in_1.stride(0),
        in_1_stride_c=in_1.stride(1),
        in_1_stride_h=in_1.stride(2),
        in_1_stride_w=in_1.stride(3),
        out_stride_b=out.stride(0),
        out_stride_c=out.stride(1),
        out_stride_h=out.stride(2),
        out_stride_w=out.stride(3),
    )
    
    return out


def replacement_func():
    return fused_adaptive_avg_pool2d_cat_kernel_wrapper