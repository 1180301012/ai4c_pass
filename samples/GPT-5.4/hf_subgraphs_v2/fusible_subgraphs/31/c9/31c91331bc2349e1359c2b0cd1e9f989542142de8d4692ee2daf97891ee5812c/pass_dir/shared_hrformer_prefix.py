import torch
import triton
import triton.language as tl


@triton.jit
def hrformer_prefix_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    rows,
    C,
    W,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    residual_stride_b,
    residual_stride_row,
    residual_stride_c,
    out_stride_b,
    out_stride_row,
    out_stride_c,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    h = pid // W
    w = pid - h * W

    x_ptrs = x_ptr + offs_c * x_stride_c + h * x_stride_h + w * x_stride_w
    residual_ptrs = residual_ptr + pid * residual_stride_row + offs_c * residual_stride_c
    out_ptrs = out_ptr + pid * out_stride_row + offs_c * out_stride_c

    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptrs, mask=mask, other=0).to(tl.float32)
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.70710678118654752440))
    y = gelu + residual
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def hrformer_prefix_route(x, residual):
    rows = residual.shape[1]
    C = residual.shape[2]
    W = x.shape[3]
    out = torch.empty_like(residual)

    if C <= 32:
        BLOCK_C = 32
        num_warps = 1
    elif C <= 128:
        BLOCK_C = 128
        num_warps = 4
    else:
        BLOCK_C = 256
        num_warps = 8

    hrformer_prefix_kernel[(rows,)](
        x,
        residual,
        out,
        rows,
        C,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out