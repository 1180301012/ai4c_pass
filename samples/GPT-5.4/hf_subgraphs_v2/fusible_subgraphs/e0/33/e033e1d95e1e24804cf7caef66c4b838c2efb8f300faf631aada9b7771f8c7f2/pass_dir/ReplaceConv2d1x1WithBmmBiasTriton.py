import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return out


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8, "BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_S": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 16, "BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 16, "BLOCK_S": 256}, num_warps=8, num_stages=2),
    ],
    key=["channels", "spatial"],
)
@triton.jit
def _bias_add_inplace_kernel(
    out_ptr,
    bias_ptr,
    stride_ob,
    stride_oc,
    stride_os,
    channels,
    spatial,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask = (offs_c[:, None] < channels) & (offs_s[None, :] < spatial)
    ptrs = out_ptr + pid_b * stride_ob + offs_c[:, None] * stride_oc + offs_s[None, :] * stride_os

    vals = tl.load(ptrs, mask=mask)
    bias = tl.load(bias_ptr + offs_c, mask=offs_c < channels, other=0.0)
    vals = vals + bias[:, None]
    tl.store(ptrs, vals, mask=mask)


def _bias_add_inplace(out, bias):
    batch = out.shape[0]
    channels = out.shape[1]
    spatial = out.shape[2]
    grid = lambda meta: (
        triton.cdiv(spatial, meta["BLOCK_S"]),
        triton.cdiv(channels, meta["BLOCK_C"]),
        batch,
    )
    _bias_add_inplace_kernel[grid](
        out,
        bias,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        channels,
        spatial,
    )


@torch.fx.wrap
def conv2d_1x1_bmm_bias_triton(bias, weight, x):
    batch = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_channels = weight.shape[0]

    x_flat = x.view(batch, in_channels, height * width)
    w_flat = weight.view(out_channels, in_channels)

    out_flat = w_flat.unsqueeze(0) @ x_flat
    _bias_add_inplace(out_flat, bias)
    out = out_flat.view(batch, out_channels, height, width)
    return out


def replacement_func():
    return conv2d_1x1_bmm_bias_triton