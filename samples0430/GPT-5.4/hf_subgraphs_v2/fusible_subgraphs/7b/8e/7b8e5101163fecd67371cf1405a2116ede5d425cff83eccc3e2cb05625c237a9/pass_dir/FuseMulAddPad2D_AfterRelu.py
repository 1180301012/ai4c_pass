import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_1, in_0):
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch._C._nn.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(tmp_2, in_1, in_0):
    return (tmp_2, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 128}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_mul_add_pad2d_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    C,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc - n * C

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_out = (offs_h[:, None] < (H + 1)) & (offs_w[None, :] < (W + 1))
    mask_in = (offs_h[:, None] < H) & (offs_w[None, :] < W)

    x_ptrs = (
        x_ptr
        + n * stride_xn
        + c * stride_xc
        + offs_h[:, None] * stride_xh
        + offs_w[None, :] * stride_xw
    )
    x = tl.load(x_ptrs, mask=mask_in, other=0.0)

    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    zero = tl.zeros((BLOCK_H, BLOCK_W), dtype=x.dtype)
    y = x * scale + bias
    out_val = tl.where(mask_in, y, zero)

    out_ptrs = (
        out_ptr
        + n * stride_on
        + c * stride_oc
        + offs_h[:, None] * stride_oh
        + offs_w[None, :] * stride_ow
    )
    tl.store(out_ptrs, out_val, mask=mask_out)


@torch.fx.wrap
def fused_mul_add_pad2d(tmp_2, in_1, in_0):
    n, c, h, w = tmp_2.shape
    out = torch.empty((n, c, h + 1, w + 1), device=tmp_2.device, dtype=tmp_2.dtype)

    grid = lambda meta: (
        triton.cdiv(w + 1, meta['BLOCK_W']),
        triton.cdiv(h + 1, meta['BLOCK_H']),
        n * c,
    )

    fused_mul_add_pad2d_kernel[grid](
        tmp_2,
        in_1,
        in_0,
        out,
        c,
        h,
        w,
        tmp_2.stride(0),
        tmp_2.stride(1),
        tmp_2.stride(2),
        tmp_2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return fused_mul_add_pad2d