import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 8}, num_warps=8, num_stages=2),
    ],
    key=["num_rows"],
)
@triton.jit
def _layernorm_transpose_gelu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_rows,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, 512)
    mask_n = offs_n < num_rows
    mask = mask_n[:, None]

    x_ptrs = x_ptr + pid_b * stride_x0 + offs_n[:, None] * stride_x1 + offs_h[None, :] * stride_x2
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    sum_x = tl.sum(xf, axis=1)
    sum_x2 = tl.sum(xf * xf, axis=1)
    mean = sum_x / 512.0
    var = sum_x2 / 512.0 - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offs_h).to(tl.float32)
    b = tl.load(bias_ptr + offs_h).to(tl.float32)

    y = (xf - mean[:, None]) * rstd[:, None]
    y = y * w[None, :] + b[None, :]
    gelu = y * (0.5 * (1.0 + tl.math.erf(y * 0.7071067811865475)))

    out_ptrs = out_ptr + pid_b * stride_o0 + offs_h[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, tl.trans(gelu), mask=mask_n[None, :])


@torch.fx.wrap
def fused_layernorm_transpose_gelu(bias, weight, x):
    batch = x.shape[0]
    num_rows = x.shape[1]
    hidden = x.shape[2]
    if hidden != 512:
        raise RuntimeError('FuseLayerNormTransposeGelu512 expects hidden size 512')

    out = torch.empty((batch, hidden, num_rows), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_N"]), batch)

    _layernorm_transpose_gelu_kernel[grid](
        x,
        weight,
        bias,
        out,
        num_rows,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1e-5,
    )

    return out


def replacement_func():
    return fused_layernorm_transpose_gelu