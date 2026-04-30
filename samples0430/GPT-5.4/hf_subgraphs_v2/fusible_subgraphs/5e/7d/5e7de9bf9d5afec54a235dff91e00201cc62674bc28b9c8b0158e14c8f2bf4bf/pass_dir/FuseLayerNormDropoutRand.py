import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    tmp_10 = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    tmp_12 = torch.rand([])
    return tmp_11


def replacement_args(x, weight, bias):
    return (x, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _layernorm_dropout_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    x_stride_b,
    x_stride_t,
    x_stride_c,
    out_stride_b,
    out_stride_t,
    out_stride_c,
    T,
    N,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // T
    t = row % T

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x_ptrs = x_ptr + b * x_stride_b + t * x_stride_t + cols * x_stride_c
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + EPS)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0).to(tl.float32)
    y = x_centered * rstd
    y = y * weight + bias

    out_ptrs = out_ptr + b * out_stride_b + t * out_stride_t + cols * out_stride_c
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_layernorm_dropout_rand(x, weight, bias):
    bsz = x.shape[0]
    tokens = x.shape[1]
    channels = x.shape[2]
    out = torch.empty((bsz, tokens, channels), device=x.device, dtype=x.dtype)

    _layernorm_dropout_kernel[(bsz * tokens,)](
        x,
        weight,
        bias,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        tokens,
        channels,
        EPS=1e-5,
    )
    return out


def replacement_func():
    return fused_layernorm_dropout_rand