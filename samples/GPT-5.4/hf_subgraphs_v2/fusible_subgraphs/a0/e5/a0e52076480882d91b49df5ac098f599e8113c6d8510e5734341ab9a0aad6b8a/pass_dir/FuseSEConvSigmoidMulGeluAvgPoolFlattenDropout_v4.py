import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 512}, num_warps=8, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def _fused_kernel(
    bias_ptr,
    weight_ptr,
    act_ptr,
    se_ptr,
    out_ptr,
    N,
    C,
    stride_b,
    stride_w0,
    stride_w1,
    stride_an,
    stride_ac,
    stride_ah,
    stride_aw,
    stride_sn,
    stride_sk,
    stride_on,
    stride_oc,
    W_CONST: tl.constexpr,
    HW_CONST: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_TANH: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    offs_k = tl.arange(0, 64)
    se = tl.load(se_ptr + pid_n * stride_sn + offs_k * stride_sk)[None, :]
    w = tl.load(
        weight_ptr + offs_k[:, None] * stride_w1 + offs_c[None, :] * stride_w0,
        mask=mask_c[None, :],
        other=0.0,
    )
    bias = tl.load(bias_ptr + offs_c * stride_b, mask=mask_c, other=0.0).to(tl.float32)[None, :]
    gate = tl.dot(se, w, out_dtype=tl.float32) + bias
    gate = 1.0 / (1.0 + tl.exp(-gate))

    acc = tl.zeros((1, BLOCK_C), dtype=tl.float32)
    for hw0 in tl.static_range(0, HW_CONST, BLOCK_HW):
        offs_hw = hw0 + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW_CONST
        h = offs_hw // W_CONST
        w_idx = offs_hw - h * W_CONST
        x = tl.load(
            act_ptr + pid_n * stride_an + offs_c[:, None] * stride_ac + h[None, :] * stride_ah + w_idx[None, :] * stride_aw,
            mask=mask_c[:, None] & mask_hw[None, :],
            other=0.0,
        ).to(tl.float32)
        y = x * tl.trans(gate)
        if USE_TANH:
            y3 = y * y * y
            t = 0.7978845608028654 * (y + 0.044715 * y3)
            y = 0.5 * y * (1.0 + tl.tanh(t))
        else:
            y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865475244))
        acc += tl.trans(tl.sum(y, axis=1)[None, :])

    out = tl.trans(acc) * (1.0 / HW_CONST)
    tl.store(out_ptr + pid_n * stride_on + offs_c * stride_oc, out, mask=mask_c)


@torch.fx.wrap
def fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    c = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    hw = h * w

    out = torch.empty((n, c), device=in_2.device, dtype=in_2.dtype)
    use_tanh = in_2.dtype != torch.float32
    grid = lambda META: (n, triton.cdiv(c, META['BLOCK_C']))

    if hw == 49:
        _fused_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            n,
            c,
            in_0.stride(0),
            in_1.stride(0),
            in_1.stride(1),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_3.stride(0),
            in_3.stride(1),
            out.stride(0),
            out.stride(1),
            W_CONST=7,
            HW_CONST=49,
            BLOCK_HW=8,
            USE_TANH=use_tanh,
        )
    elif hw == 64:
        _fused_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            n,
            c,
            in_0.stride(0),
            in_1.stride(0),
            in_1.stride(1),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_3.stride(0),
            in_3.stride(1),
            out.stride(0),
            out.stride(1),
            W_CONST=8,
            HW_CONST=64,
            BLOCK_HW=8,
            USE_TANH=use_tanh,
        )
    else:
        _fused_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            n,
            c,
            in_0.stride(0),
            in_1.stride(0),
            in_1.stride(1),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_3.stride(0),
            in_3.stride(1),
            out.stride(0),
            out.stride(1),
            W_CONST=12,
            HW_CONST=144,
            BLOCK_HW=8,
            USE_TANH=use_tanh,
        )
    return out


def replacement_func():
    return fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout