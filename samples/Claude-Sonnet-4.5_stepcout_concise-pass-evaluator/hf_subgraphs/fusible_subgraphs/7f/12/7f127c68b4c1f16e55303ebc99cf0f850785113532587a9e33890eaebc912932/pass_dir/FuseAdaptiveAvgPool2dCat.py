import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match: adaptive_avg_pool2d followed by cat"""
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['total_out_elements'],
)
@triton.jit
def fused_pool_cat_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    batch_size,
    in0_channels,
    in1_channels,
    in0_height,
    in0_width,
    out_height,
    out_width,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    total_out_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused kernel for adaptive_avg_pool2d + concat with autotuning.
    """
    pid = tl.program_id(0)
    
    # Each thread handles BLOCK_SIZE elements
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_out_elements
    
    # Calculate output dimensions
    out_channels = in0_channels + in1_channels
    spatial_size = out_height * out_width
    
    # Decompose linear index into (batch, channel, h, w)
    w = idx % out_width
    h = (idx // out_width) % out_height
    c = (idx // spatial_size) % out_channels
    b = idx // (out_channels * spatial_size)
    
    # Determine if this element comes from in0 (pooled) or in1 (direct)
    from_in0 = c < in0_channels
    
    # Initialize output value
    out_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process in0 elements (with pooling)
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for dh in range(stride_h):
        for dw in range(stride_w):
            src_h = h * stride_h + dh
            src_w = w * stride_w + dw
            
            # Build input index for in0
            in0_idx = (b * in0_channels * in0_height * in0_width +
                      c * in0_height * in0_width +
                      src_h * in0_width +
                      src_w)
            
            # Load with mask (only if from_in0 and within bounds)
            in0_mask = mask & from_in0 & (src_h < in0_height) & (src_w < in0_width)
            val = tl.load(in0_ptr + in0_idx, mask=in0_mask, other=0.0)
            pool_sum += val
    
    # Average the pooled values
    pool_avg = pool_sum / (stride_h * stride_w)
    out_val = tl.where(from_in0, pool_avg, out_val)
    
    # Process in1 elements (direct copy)
    c_in1 = c - in0_channels
    in1_idx = (b * in1_channels * spatial_size +
              c_in1 * spatial_size +
              h * out_width +
              w)
    
    in1_mask = mask & ~from_in0
    val = tl.load(in1_ptr + in1_idx, mask=in1_mask, other=0.0)
    out_val = tl.where(~from_in0, val, out_val)
    
    # Store result
    tl.store(out_ptr + idx, out_val, mask=mask)


@torch.fx.wrap
def fused_adaptive_pool_cat(in_0, in_1):
    """
    Fused implementation of adaptive_avg_pool2d + cat.
    """
    batch_size, in0_channels, in0_height, in0_width = in_0.shape
    _, in1_channels, out_height, out_width = in_1.shape
    
    # Calculate pooling strides
    stride_h = in0_height // out_height
    stride_w = in0_width // out_width
    
    # Output shape: [batch_size, in0_channels + in1_channels, out_height, out_width]
    out = torch.empty(
        (batch_size, in0_channels + in1_channels, out_height, out_width),
        device=in_0.device,
        dtype=in_0.dtype,
    )
    
    # Launch kernel
    total_elements = out.numel()
    grid = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_pool_cat_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        in0_channels,
        in1_channels,
        in0_height,
        in0_width,
        out_height,
        out_width,
        stride_h=stride_h,
        stride_w=stride_w,
        total_out_elements=total_elements,
    )
    
    return out


def replacement_func():
    return fused_adaptive_pool_cat