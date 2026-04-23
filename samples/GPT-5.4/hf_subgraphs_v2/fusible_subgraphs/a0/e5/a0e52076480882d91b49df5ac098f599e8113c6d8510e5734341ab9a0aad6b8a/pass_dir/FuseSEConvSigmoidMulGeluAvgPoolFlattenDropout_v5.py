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
        triton.Config({'BLOCK_N': 1, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_C': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_C': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_C': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_C': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_C': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def _se_conv_sigmoid_mul_gelu_avgpool_kernel(
    bias_ptr,
    weight_ptr,
    act_ptr,
    se_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    HW,
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
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C
    mask_nc = mask_n[:, None] & mask_c[None, :]

    bias = tl.load(bias_ptr + offs_c * stride_b, mask=mask_c, other=0.0).to(tl.float32)
    gate_acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32) + bias[None, :]

    for k in tl.static_range(0, 64, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        se_tile = tl.load(
            se_ptr + offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk,
            mask=mask_n[:, None] & (offs_k[None, :] < 64),
            other=0.0,
        )
        w_tile = tl.load(
            weight_ptr + offs_k[:, None] * stride_w1 + offs_c[None, :] * stride_w0,
            mask=(offs_k[:, None] < 64) & mask_c[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(se_tile, w_tile, out_dtype=tl.float32)

    gate = 1.0 / (1.0 + tl.exp(-gate_acc))

    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    for hw in tl.range(0, HW):
        h = hw // W
        w = hw - h * W
        x = tl.load(
            act_ptr + offs_n[:, None] * stride_an + offs_c[None, :] * stride_ac + h * stride_ah + w * stride_aw,
            mask=mask_nc,
            other=0.0,
        ).to(tl.float32)
        y = x * gate
        y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865475244))
        acc += y

    out = acc * (1.0 / HW)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_c[None, :] * stride_oc, out, mask=mask_nc)


@torch.fx.wrap
def fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    c = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    hw = h * w

    out = torch.empty((n, c), device=in_2.device, dtype=in_2.dtype)

    grid = lambda META: (triton.cdiv(n, META['BLOCK_N']), triton.cdiv(c, META['BLOCK_C']))
    _se_conv_sigmoid_mul_gelu_avgpool_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        n,
        c,
        h,
        w,
        hw,
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
    )
    return out


def replacement_func():
    return fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout