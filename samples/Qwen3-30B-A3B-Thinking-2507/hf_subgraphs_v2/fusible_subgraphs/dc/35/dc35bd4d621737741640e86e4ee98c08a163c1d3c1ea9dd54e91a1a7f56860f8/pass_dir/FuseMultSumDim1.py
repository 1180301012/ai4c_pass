import torch
import triton
import triton.language as tl

# Pattern matching function for the fused multiply-sum operation
# Matches: tmp_0 = in_1 * in_0; tmp_1 = torch.sum(tmp_0, dim=1)
def pattern(x, y):
    z = x * y
    return torch.sum(z, dim=1)

# Argument extraction - returns inputs to the product (in_0, in_1)
def replacement_args(x, y):
    return (x, y)

# Triton kernel for fused multiply-sum along dim=1
@triton.jit
def fused_mult_sum_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B, C, H, W,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    out_stride0, out_stride1, out_stride2,
    BLOCK_C: tl.constexpr,
):
    # Output index: (b, h, w)
    b_idx = tl.program_id(0) // (H * W)
    h_idx = (tl.program_id(0) // W) % H
    w_idx = tl.program_id(0) % W

    # Calculate output pointer
    out_ptr += b_idx * out_stride0 + h_idx * out_stride1 + w_idx * out_stride2

    # Initialize sum accumulator
    sum_val = tl.zeros((), dtype=tl.float32)

    # Process channels in blocks of size BLOCK_C
    for c_start in range(0, C, BLOCK_C):
        c_end = min(c_start + BLOCK_C, C)
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask = c_offsets < C

        # Load x and y values for this channel block
        x_vals = tl.load(
            x_ptr + b_idx * x_stride0 + c_offsets * x_stride1 + h_idx * x_stride2 + w_idx * x_stride3,
            mask=mask,
            other=0.0
        )
        y_vals = tl.load(
            y_ptr + b_idx * y_stride0 + c_offsets * y_stride1 + h_idx * y_stride2 + w_idx * y_stride3,
            mask=mask,
            other=0.0
        )
        products = x_vals * y_vals
        sum_val += tl.sum(products)

    # Store the result
    tl.store(out_ptr, sum_val)

# Wrapper for Triton kernel
@torch.fx.wrap
def fused_mult_sum(x, y):
    # Expected input shapes: [B, C, H, W]
    B, C, H, W = x.shape
    out = torch.empty([B, H, W], dtype=x.dtype, device=x.device)

    # Compute strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    y_stride0, y_stride1, y_stride2, y_stride3 = y.stride()
    out_stride0, out_stride1, out_stride2 = out.stride()

    # Grid dimensions: one thread per output element
    grid = (B * H * W,)
    BLOCK_C = 64

    # Launch kernel
    fused_mult_sum_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        x_stride0=x_stride0, x_stride1=x_stride1, x_stride2=x_stride2, x_stride3=x_stride3,
        y_stride0=y_stride0, y_stride1=y_stride1, y_stride2=y_stride2, y_stride3=y_stride3,
        out_stride0=out_stride0, out_stride1=out_stride1, out_stride2=out_stride2,
        BLOCK_C=BLOCK_C
    )

    return out

def replacement_func():
    return fused_mult_sum