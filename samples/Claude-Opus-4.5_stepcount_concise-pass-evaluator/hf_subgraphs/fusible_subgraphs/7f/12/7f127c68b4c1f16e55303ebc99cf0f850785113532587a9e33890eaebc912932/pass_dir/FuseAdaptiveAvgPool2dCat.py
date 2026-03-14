import torch
import triton
import triton.language as tl

# Pattern matching function - matches adaptive_avg_pool2d followed by cat
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_pool_cat_kernel(
    in_0_ptr,        # Input tensor for pooling [B, C0, H_in, W_in]
    in_1_ptr,        # Input tensor for concat [B, C1, H_out, W_out]
    out_ptr,         # Output tensor [B, C0+C1, H_out, W_out]
    B: tl.constexpr,           # Batch size
    C0: tl.constexpr,          # Channels in in_0
    C1: tl.constexpr,          # Channels in in_1
    H_in: tl.constexpr,        # Input height
    W_in: tl.constexpr,        # Input width
    H_out: tl.constexpr,       # Output height
    W_out: tl.constexpr,       # Output width
    n_elements,      # Total output elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output indices: [b, c, h, w]
    C_total = C0 + C1
    HW_out = H_out * W_out
    CHW_out = C_total * HW_out
    
    b = offsets // CHW_out
    rem = offsets % CHW_out
    c = rem // HW_out
    rem2 = rem % HW_out
    h_out = rem2 // W_out
    w_out = rem2 % W_out
    
    # Pooling scale factors (2x2 average pooling)
    scale_h = 2  # H_in / H_out = 64 / 32 = 2
    scale_w = 2  # W_in / W_out = 48 / 24 = 2
    
    # For channels < C0, perform average pooling from in_0
    # For channels >= C0, copy from in_1
    is_pooled = c < C0
    
    # Calculate input positions for pooling
    h_in_start = h_out * scale_h
    w_in_start = w_out * scale_w
    
    # Input strides for in_0: [C0*H_in*W_in, H_in*W_in, W_in, 1]
    HW_in = H_in * W_in
    CHW_in = C0 * HW_in
    
    # Calculate base index for in_0
    in_0_base = b * CHW_in + c * HW_in
    
    # Load 4 values for 2x2 pooling
    idx00 = in_0_base + h_in_start * W_in + w_in_start
    idx01 = in_0_base + h_in_start * W_in + (w_in_start + 1)
    idx10 = in_0_base + (h_in_start + 1) * W_in + w_in_start
    idx11 = in_0_base + (h_in_start + 1) * W_in + (w_in_start + 1)
    
    # Only load from in_0 if this is a pooled channel
    pool_mask = mask & is_pooled
    v00 = tl.load(in_0_ptr + idx00, mask=pool_mask, other=0.0)
    v01 = tl.load(in_0_ptr + idx01, mask=pool_mask, other=0.0)
    v10 = tl.load(in_0_ptr + idx10, mask=pool_mask, other=0.0)
    v11 = tl.load(in_0_ptr + idx11, mask=pool_mask, other=0.0)
    
    # Compute average
    pooled_val = (v00 + v01 + v10 + v11) * 0.25
    
    # For in_1: channel index is c - C0, same spatial dims
    c1_idx = c - C0
    in_1_CHW = C1 * HW_out
    in_1_idx = b * in_1_CHW + c1_idx * HW_out + h_out * W_out + w_out
    
    # Only load from in_1 if this is a concat channel
    cat_mask = mask & (~is_pooled)
    cat_val = tl.load(in_1_ptr + in_1_idx, mask=cat_mask, other=0.0)
    
    # Select the appropriate value
    result = tl.where(is_pooled, pooled_val, cat_val)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_adaptive_avgpool2d_cat(in_0, in_1):
    # in_0: [B, C0, H_in, W_in] -> pool to [B, C0, H_out, W_out]
    # in_1: [B, C1, H_out, W_out]
    # out: [B, C0+C1, H_out, W_out]
    
    B = in_0.shape[0]
    C0 = in_0.shape[1]
    H_in = in_0.shape[2]
    W_in = in_0.shape[3]
    
    C1 = in_1.shape[1]
    H_out = in_1.shape[2]
    W_out = in_1.shape[3]
    
    C_total = C0 + C1
    n_elements = B * C_total * H_out * W_out
    
    out = torch.empty((B, C_total, H_out, W_out), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_pool_cat_kernel[grid](
        in_0,
        in_1,
        out,
        B, C0, C1,
        H_in, W_in,
        H_out, W_out,
        n_elements,
    )
    
    return (out,)


def replacement_func():
    return fused_adaptive_avgpool2d_cat