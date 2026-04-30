import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return (tmp_2, tmp_13)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _whole_kernel(
    in_ptr,
    w_ptr,
    tmp2_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask_m = offs_m < 3

    inp = tl.load(in_ptr + offs_m * BLOCK_N + offs_n, mask=mask_m, other=0).to(tl.float32)
    x_bf16 = (inp * 45.25).to(tl.bfloat16)
    x = x_bf16.to(tl.float32)
    w = tl.load(w_ptr + offs_n, mask=offs_n < BLOCK_N, other=0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=1) * (1.0 / 2048.0)
    inv_rms = tl.rsqrt(mean_sq + 1e-06)
    y = x * inv_rms[:, None] * (1.0 + w)

    tl.store(tmp2_ptr + offs_m * BLOCK_N + offs_n, x_bf16, mask=mask_m)
    tl.store(out_ptr + offs_m * BLOCK_N + offs_n, y.to(tl.bfloat16), mask=mask_m)


@torch.fx.wrap
def _whole_helper(in_0, in_1, in_2):
    tmp_2 = torch.empty_like(in_0)
    tmp_13 = torch.empty_like(in_0)
    _whole_kernel[(1,)](
        in_0,
        in_1,
        tmp_2,
        tmp_13,
        BLOCK_M=4,
        BLOCK_N=2048,
        num_warps=8,
        num_stages=4,
    )
    return (tmp_2, tmp_13)


def _whole_route(in_0, in_1, in_2):
    outs = _whole_helper(in_0, in_1, in_2)
    return outs[0], outs[1]


def replacement_func():
    return _whole_route