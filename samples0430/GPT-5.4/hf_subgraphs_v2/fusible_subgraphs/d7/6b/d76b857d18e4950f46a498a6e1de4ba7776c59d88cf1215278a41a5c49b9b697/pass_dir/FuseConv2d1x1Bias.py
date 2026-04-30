import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
    ],
    key=['total_rows', 'c_in', 'c_out'],
)
@triton.jit
def conv1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    total_rows,
    height,
    width,
    c_in,
    c_out,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wi,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    hw = offs_m % (height * width)
    n_idx = offs_m // (height * width)
    h_idx = hw // width
    w_idx = hw % width

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, c_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + offs_k[None, :] * stride_xc
            + h_idx[:, None] * stride_xh
            + w_idx[:, None] * stride_xw
        )
        x_mask = (offs_m[:, None] < total_rows) & (offs_k[None, :] < c_in)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wo + offs_k[:, None] * stride_wi
        w_mask = (offs_n[None, :] < c_out) & (offs_k[:, None] < c_in)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < c_out, other=0.0)
    acc += bias[None, :]

    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
    )
    out_mask = (offs_m[:, None] < total_rows) & (offs_n[None, :] < c_out)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def triton_conv2d_1x1_bias(bias, weight, inp):
    batch = inp.shape[0]
    c_out = weight.shape[0]
    height = inp.shape[2]
    width = inp.shape[3]
    c_in = inp.shape[1]
    total_rows = batch * height * width

    out = torch.empty((batch, c_out, height, width), device=inp.device, dtype=inp.dtype)

    grid = lambda meta: (triton.cdiv(total_rows, meta['BLOCK_M']),)
    conv1x1_nchw_kernel[grid](
        inp,
        weight,
        bias,
        out,
        total_rows,
        height,
        width,
        c_in,
        c_out,
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        inp.stride(3),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return triton_conv2d_1x1_bias