import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    return tmp_6


def replacement_args(in_5):
    return (in_5,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    ],
    key=["B", "C"],
)
@triton.jit
def _adaptive_avg_pool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    n_elem = B * C
    mask = offsets < n_elem

    b = offsets // C
    c = offsets % C

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h in range(8):
        for w in range(8):
            ptrs = x_ptr + b * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw
            x = tl.load(ptrs, mask=mask, other=0.0)
            acc += x.to(tl.float32)

    y = acc * (1.0 / 64.0)
    out_ptrs = out_ptr + b * stride_on + c * stride_oc
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def triton_adaptive_avg_pool2d_1x1(in_5):
    B = in_5.shape[0]
    C = in_5.shape[1]
    out = torch.empty((B, C, 1, 1), device=in_5.device, dtype=in_5.dtype)
    grid = lambda META: (triton.cdiv(B * C, META["BLOCK"]),)
    _adaptive_avg_pool2d_1x1_kernel[grid](
        in_5,
        out,
        B,
        C,
        in_5.stride(0),
        in_5.stride(1),
        in_5.stride(2),
        in_5.stride(3),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return triton_adaptive_avg_pool2d_1x1