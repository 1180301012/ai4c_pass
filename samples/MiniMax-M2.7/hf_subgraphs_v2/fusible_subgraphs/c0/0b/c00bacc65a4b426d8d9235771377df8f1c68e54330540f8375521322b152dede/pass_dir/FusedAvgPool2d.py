import torch
import triton
import triton.language as tl

"""
Fused AvgPool2d kernel for GPU optimization.

This pass optimizes the avg_pool2d operation with a custom Triton kernel.
The kernel is optimized for common pool sizes (2x2 with stride 2).
"""

def pattern(in_6):
    """
    Match avg_pool2d pattern.
    
    Args:
        in_6: Input tensor [N, C, H, W]
    
    Returns:
        Pooled tensor [N, C, H/2, W/2]
    """
    tmp_7 = torch.nn.functional.avg_pool2d(in_6, 2, 2, 0, True, False, None)
    return tmp_7

def replacement_args(in_6):
    return (in_6,)

@triton.jit
def avg_pool2d_kernel(
    x_ptr, output_ptr,
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    dtype: tl.constexpr
):
    """
    AvgPool2d kernel with 2x2 window, stride 2, no padding.
    
    Each program computes one output element (n, c, h_out, w_out).
    """
    pid = tl.program_id(0)
    
    # Decode output position
    n = pid // (C * (H // 2) * (W // 2))
    tmp = pid % (C * (H // 2) * (W // 2))
    c = tmp // ((H // 2) * (W // 2))
    tmp = tmp % ((H // 2) * (W // 2))
    h_out = tmp // (W // 2)
    w_out = tmp % (W // 2)
    
    # Input region for 2x2 pooling (h_base, w_base is top-left corner)
    h_base = h_out * 2
    w_base = w_out * 2
    
    # Compute all 4 linear indices
    base_idx = n * stride_x_n + c * stride_x_c
    
    # Load values using strided access
    idx0 = base_idx + h_base * stride_x_h + w_base * stride_x_w
    idx1 = base_idx + (h_base + 1) * stride_x_h + w_base * stride_x_w
    idx2 = base_idx + h_base * stride_x_h + (w_base + 1) * stride_x_w
    idx3 = base_idx + (h_base + 1) * stride_x_h + (w_base + 1) * stride_x_w
    
    # Load all 4 values
    val0 = tl.load(x_ptr + idx0)
    val1 = tl.load(x_ptr + idx1)
    val2 = tl.load(x_ptr + idx2)
    val3 = tl.load(x_ptr + idx3)
    
    # Sum all 4 values
    sum_val = val0 + val1 + val2 + val3
    
    # Compute average (count is always 4 for 2x2 pool with no padding)
    avg_val = sum_val / tl.cast(4, dtype)
    
    # Store result
    out_idx = n * stride_out_n + c * stride_out_c + h_out * stride_out_h + w_out * stride_out_w
    tl.store(output_ptr + out_idx, avg_val)

@torch.fx.wrap
def fused_avg_pool2d_impl(x):
    """Wrapper for the avg_pool2d kernel."""
    N, C, H, W = x.shape
    H_out, W_out = H // 2, W // 2
    
    output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    total_elements = N * C * H_out * W_out
    
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Determine dtype for the kernel
    if x.dtype == torch.float16:
        dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float32
    
    avg_pool2d_kernel[grid](
        x, output,
        N, C, H, W,
        x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[3],
        output.stride()[0], output.stride()[1], output.stride()[2], output.stride()[3],
        dtype=dtype
    )
    return output

def replacement_func():
    return fused_avg_pool2d_impl