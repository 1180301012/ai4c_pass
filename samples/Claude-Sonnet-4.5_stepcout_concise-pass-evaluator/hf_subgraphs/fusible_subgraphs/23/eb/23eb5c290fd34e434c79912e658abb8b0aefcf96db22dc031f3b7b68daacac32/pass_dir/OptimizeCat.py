import torch
import triton
import triton.language as tl


def pattern(a, b):
    """
    Simple pattern: just cat
    """
    result = torch.cat([a, b], dim=-1)
    return result


def replacement_args(a, b):
    return (a, b)


@triton.jit
def concat_kernel(
    a_ptr, b_ptr, out_ptr,
    B, H, W, N_a, N_b,
    stride_a_b, stride_a_h, stride_a_w, stride_a_n,
    stride_b_b, stride_b_h, stride_b_w, stride_b_n,
    stride_out_b, stride_out_h, stride_out_w, stride_out_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_h = pid_hw // W
    pid_w = pid_hw % W
    
    # Base offsets
    base_a = pid_b * stride_a_b + pid_h * stride_a_h + pid_w * stride_a_w
    base_b = pid_b * stride_b_b + pid_h * stride_b_h + pid_w * stride_b_w
    base_out = pid_b * stride_out_b + pid_h * stride_out_h + pid_w * stride_out_w
    
    # Copy first part (a)
    for i in range(0, N_a, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_a
        vals = tl.load(a_ptr + base_a + offsets * stride_a_n, mask=mask)
        tl.store(out_ptr + base_out + offsets * stride_out_n, vals, mask=mask)
    
    # Copy second part (b)
    for i in range(0, N_b, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_b
        vals = tl.load(b_ptr + base_b + offsets * stride_b_n, mask=mask)
        tl.store(out_ptr + base_out + (N_a + offsets) * stride_out_n, vals, mask=mask)


@torch.fx.wrap
def concat_impl(a, b):
    B, H, W, N_a = a.shape
    N_b = b.shape[3]
    
    out = torch.empty((B, H, W, N_a + N_b), device=a.device, dtype=a.dtype)
    
    grid = (B, H * W)
    BLOCK_SIZE = 64
    
    concat_kernel[grid](
        a, b, out,
        B, H, W, N_a, N_b,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return concat_impl