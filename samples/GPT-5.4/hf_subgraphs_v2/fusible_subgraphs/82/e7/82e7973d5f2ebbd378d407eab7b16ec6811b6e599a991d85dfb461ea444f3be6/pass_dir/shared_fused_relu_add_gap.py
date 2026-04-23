import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_add_gap_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    out_stride_n,
    out_stride_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    n = pid // C

    hw = H * W
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < hw

    h_idx = offs // W
    w_idx = offs % W

    x_base = x_ptr + n * x_stride_n + c * x_stride_c
    y_base = y_ptr + n * y_stride_n + c * y_stride_c

    x_vals = tl.load(x_base + h_idx * x_stride_h + w_idx * x_stride_w, mask=mask, other=0.0)
    y_vals = tl.load(y_base + h_idx * y_stride_h + w_idx * y_stride_w, mask=mask, other=0.0)

    vals = x_vals + tl.maximum(y_vals, 0)
    acc = tl.sum(vals.to(tl.float32), axis=0)
    avg = acc / hw

    out_base = out_ptr + n * out_stride_n + c * out_stride_c
    tl.store(out_base, avg)


@torch.fx.wrap
def fused_relu_add_gap_dispatch(in_0, in_1, route):
    # route is intentionally ignored; it exists so multiple pass files can
    # share exactly the same replacement function object while differentiating
    # match patterns through replacement_args.
    N = in_0.shape[0]
    C = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]

    out = torch.empty((N, C, 1, 1), device=in_0.device, dtype=in_0.dtype)

    grid = (N * C,)
    fused_relu_add_gap_kernel[grid](
        in_0,
        in_1,
        out,
        N,
        C,
        H,
        W,
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
        BLOCK_HW=256,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_relu_add_gap_dispatch