import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64, "BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_C": 256, "BLOCK_HW": 256}, num_warps=8),
    ],
    key=["C", "HW"],
)
@triton.jit
def _fused_silu_gap_kernel(
    x_ptr,
    out_ptr,
    C,
    H,
    W,
    HW,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.arange(0, BLOCK_HW)
    c_mask = c_offsets < C
    hw_mask = hw_offsets < HW
    mask = c_mask[:, None] & hw_mask[None, :]

    h = hw_offsets // W
    w = hw_offsets % W
    ptrs = (
        x_ptr
        + pid_n * stride_n
        + c_offsets[:, None] * stride_c
        + h[None, :] * stride_h
        + w[None, :] * stride_w
    )

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    silu = x / (1.0 + tl.exp(-x))
    tl.store(ptrs, silu, mask=mask)

    acc = tl.sum(silu, axis=1)
    out = acc / HW
    out_ptrs = out_ptr + pid_n * out_stride_n + c_offsets * out_stride_c
    tl.store(out_ptrs, out, mask=c_mask)


@torch.fx.wrap
def fused_silu_adaptive_avg_pool2d_flatten_dropout_dispatch(x, route):
    # route is intentionally unused; it lets several pass files share one exact replacement function object.
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(C, META["BLOCK_C"]), N)
    _fused_silu_gap_kernel[grid](
        x,
        out,
        C,
        H,
        W,
        HW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_silu_adaptive_avg_pool2d_flatten_dropout_dispatch