import math
import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches:
#   slice -> gelu -> transpose -> add -> dropout(eval) -> layer_norm
# Returns both observable outputs from the original graph.
def pattern(in_0, in_1, in_3, conv1d_out, dropout_p):
    tmp_4 = conv1d_out[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, dropout_p, False, False)
    tmp_10 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_10)


def replacement_args(in_0, in_1, in_3, conv1d_out, dropout_p):
    return (conv1d_out, in_3, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=[],
)
@triton.jit
def fused_post_conv_kernel(
    conv_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_ln_ptr,
    batch_size,
    seq_len,
    hidden_size,
    conv_stride_b,
    conv_stride_c,
    conv_stride_l,
    residual_stride_b,
    residual_stride_l,
    residual_stride_c,
    out_add_stride_b,
    out_add_stride_l,
    out_add_stride_c,
    out_ln_stride_b,
    out_ln_stride_l,
    out_ln_stride_c,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = batch_size * seq_len
    if pid >= rows:
        return

    b = pid // seq_len
    pos = pid - b * seq_len

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    conv_offsets = (
        b * conv_stride_b
        + offs * conv_stride_c
        + pos * conv_stride_l
    )
    res_offsets = (
        b * residual_stride_b
        + pos * residual_stride_l
        + offs * residual_stride_c
    )

    x = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0)
    r = tl.load(residual_ptr + res_offsets, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    r_fp32 = r.to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu = 0.5 * x_fp32 * (1.0 + tl.erf(x_fp32 * 0.7071067811865475))
    add_out = gelu + r_fp32

    out_add_offsets = (
        b * out_add_stride_b
        + pos * out_add_stride_l
        + offs * out_add_stride_c
    )
    tl.store(out_add_ptr + out_add_offsets, add_out, mask=mask)

    mean = tl.sum(add_out, axis=0) / hidden_size
    centered = add_out - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b0 = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ln_out = centered * inv_std * w + b0

    out_ln_offsets = (
        b * out_ln_stride_b
        + pos * out_ln_stride_l
        + offs * out_ln_stride_c
    )
    tl.store(out_ln_ptr + out_ln_offsets, ln_out, mask=mask)


@torch.fx.wrap
def fused_post_conv_gelu_transpose_add_dropout_layernorm(conv1d_out, residual, weight, bias):
    batch_size = residual.shape[0]
    seq_len = residual.shape[1]
    hidden_size = residual.shape[2]

    out_add = torch.empty_like(residual)
    out_ln = torch.empty_like(residual)

    grid = (batch_size * seq_len,)
    fused_post_conv_kernel[grid](
        conv1d_out,
        residual,
        weight,
        bias,
        out_add,
        out_ln,
        batch_size,
        seq_len,
        hidden_size,
        conv1d_out.stride(0),
        conv1d_out.stride(1),
        conv1d_out.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out_add.stride(0),
        out_add.stride(1),
        out_add.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        1e-05,
        BLOCK_SIZE=1024,
    )
    return (out_add, out_ln)


def replacement_func():
    return fused_post_conv_gelu_transpose_add_dropout_layernorm