import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_scale_softmax_kernel_2d(
    x_ptr, out_ptr,
    scale,
    stride_bh: tl.constexpr, stride_m: tl.constexpr,
    N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    x_row_offset = pid_bh * stride_bh + pid_m * stride_m
    out_row_offset = pid_bh * stride_bh + pid_m * stride_m

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(x_ptr + x_row_offset + offsets, mask=mask, other=-float('inf')).to(tl.float32)
    x = x * scale

    row_max = tl.max(x, axis=0)
    x = x - row_max
    numerator = tl.exp(x)
    row_sum = tl.sum(numerator, axis=0)
    result = numerator / row_sum

    tl.store(out_ptr + out_row_offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(x):
    B, H, M, N = x.shape
    scale = 0.1767766952966369

    out = torch.empty(B, H, M, N, dtype=x.dtype, device=x.device)

    stride_bh = M * N
    stride_m = N
    BLOCK_N = 512

    fused_scale_softmax_kernel_2d[(M, B * H)](
        x_ptr=x, out_ptr=out,
        scale=scale,
        stride_bh=stride_bh, stride_m=stride_m,
        N=N, BLOCK_N=BLOCK_N,
    )

    result = out.transpose(-2, -1)
    return result


def replacement_func():
    return fused_scale_softmax_transpose