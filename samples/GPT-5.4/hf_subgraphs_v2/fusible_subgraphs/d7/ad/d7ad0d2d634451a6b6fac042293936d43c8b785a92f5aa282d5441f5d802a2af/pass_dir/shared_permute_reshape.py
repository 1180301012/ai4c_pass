import torch
import triton
import triton.language as tl


@triton.jit
def _permute_reshape_kernel(
    x_ptr,
    out_ptr,
    x_stride_b,
    x_stride_hw,
    x_stride_c,
    out_stride_b,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    B,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * C * 16 * 16
    mask = offs < total

    w = offs % 16
    t0 = offs // 16
    h = t0 % 16
    t1 = t0 // 16
    c = t1 % C
    b = t1 // C

    hw = h * 16 + w

    x_ptrs = x_ptr + b * x_stride_b + hw * x_stride_hw + c * x_stride_c
    vals = tl.load(x_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + b * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptrs, vals, mask=mask)


@torch.fx.wrap
def shared_permute_reshape_dispatch(x, route):
    b = x.shape[0]
    c = x.shape[2]

    if route == "b2" and b != 2:
        raise RuntimeError(f"Expected batch 2, got {b}")
    if route == "b8" and b != 8:
        raise RuntimeError(f"Expected batch 8, got {b}")
    if route == "b12" and b != 12:
        raise RuntimeError(f"Expected batch 12, got {b}")
    if route == "b24" and b != 24:
        raise RuntimeError(f"Expected batch 24, got {b}")
    if route == "b64" and b != 64:
        raise RuntimeError(f"Expected batch 64, got {b}")

    out = torch.empty((b, c, 16, 16), device=x.device, dtype=x.dtype)

    x_stride_b, x_stride_hw, x_stride_c = x.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()

    total = b * c * 16 * 16
    grid = (triton.cdiv(total, 1024),)
    _permute_reshape_kernel[grid](
        x,
        out,
        x_stride_b,
        x_stride_hw,
        x_stride_c,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        b,
        c,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=2,
    )
    return out