import torch
import triton
import triton.language as tl


BLOCK_HW_CONST = 8
MAX_SPLITS_CONST = 18


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
        triton.Config({'BLOCK_N': 1, 'BLOCK_C': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_C': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 4, 'BLOCK_C': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 4, 'BLOCK_C': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 8, 'BLOCK_C': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def _gate_kernel(
    bias_ptr,
    weight_ptr,
    se_ptr,
    gate_ptr,
    N,
    C,
    stride_b,
    stride_w0,
    stride_w1,
    stride_sn,
    stride_sk,
    stride_gn,
    stride_gc,
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

    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    acc += tl.load(bias_ptr + offs_c * stride_b, mask=mask_c, other=0.0).to(tl.float32)[None, :]

    for k in tl.static_range(0, 64, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        se = tl.load(
            se_ptr + offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk,
            mask=mask_n[:, None] & (offs_k[None, :] < 64),
            other=0.0,
        )
        w = tl.load(
            weight_ptr + offs_k[:, None] * stride_w1 + offs_c[None, :] * stride_w0,
            mask=(offs_k[:, None] < 64) & mask_c[None, :],
            other=0.0,
        )
        acc += tl.dot(se, w, out_dtype=tl.float32)

    gate = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(gate_ptr + offs_n[:, None] * stride_gn + offs_c[None, :] * stride_gc, gate, mask=mask_nc)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _partial_pool_kernel(
    act_ptr,
    gate_ptr,
    partial_ptr,
    N,
    C,
    W,
    HW,
    NUM_SPLITS,
    stride_an,
    stride_ac,
    stride_ah,
    stride_aw,
    stride_gn,
    stride_gc,
    stride_pn,
    stride_ps,
    stride_pc,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_ns = tl.program_id(1)

    pid_n = pid_ns // NUM_SPLITS
    pid_s = pid_ns - pid_n * NUM_SPLITS

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_s * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_c = offs_c < C
    mask_hw = offs_hw < HW
    valid_n = pid_n < N

    gate = tl.load(gate_ptr + pid_n * stride_gn + offs_c * stride_gc, mask=valid_n & mask_c, other=0.0).to(tl.float32)

    h = offs_hw // W
    w = offs_hw - h * W
    ptrs = act_ptr + pid_n * stride_an + offs_c[:, None] * stride_ac + h[None, :] * stride_ah + w[None, :] * stride_aw
    x = tl.load(ptrs, mask=valid_n & mask_c[:, None] & mask_hw[None, :], other=0.0).to(tl.float32)
    y = x * gate[:, None]
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865475244))
    acc = tl.sum(y, axis=1)

    tl.store(partial_ptr + pid_n * stride_pn + pid_s * stride_ps + offs_c * stride_pc, acc, mask=valid_n & mask_c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def _finalize_kernel(
    partial_ptr,
    out_ptr,
    N,
    C,
    HW,
    NUM_SPLITS,
    stride_pn,
    stride_ps,
    stride_pc,
    stride_on,
    stride_oc,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    valid_n = pid_n < N

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for s in tl.static_range(0, 18):
        vals = tl.load(
            partial_ptr + pid_n * stride_pn + s * stride_ps + offs_c * stride_pc,
            mask=valid_n & mask_c & (s < NUM_SPLITS),
            other=0.0,
        ).to(tl.float32)
        acc += vals

    out = acc * (1.0 / HW)
    tl.store(out_ptr + pid_n * stride_on + offs_c * stride_oc, out, mask=valid_n & mask_c)


@torch.fx.wrap
def fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    c = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    hw = h * w
    num_splits = triton.cdiv(hw, BLOCK_HW_CONST)

    gate = torch.empty((n, c), device=in_2.device, dtype=torch.float32)
    partial = torch.empty((n, MAX_SPLITS_CONST, c), device=in_2.device, dtype=torch.float32)
    out = torch.empty((n, c), device=in_2.device, dtype=in_2.dtype)

    grid_gate = lambda META: (triton.cdiv(n, META['BLOCK_N']), triton.cdiv(c, META['BLOCK_C']))
    _gate_kernel[grid_gate](
        in_0,
        in_1,
        in_3,
        gate,
        n,
        c,
        in_0.stride(0),
        in_1.stride(0),
        in_1.stride(1),
        in_3.stride(0),
        in_3.stride(1),
        gate.stride(0),
        gate.stride(1),
    )

    grid_partial = lambda META: (triton.cdiv(c, META['BLOCK_C']), n * num_splits)
    _partial_pool_kernel[grid_partial](
        in_2,
        gate,
        partial,
        n,
        c,
        w,
        hw,
        num_splits,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        gate.stride(0),
        gate.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        BLOCK_HW=BLOCK_HW_CONST,
    )

    grid_final = lambda META: (triton.cdiv(c, META['BLOCK_C']), n)
    _finalize_kernel[grid_final](
        partial,
        out,
        n,
        c,
        hw,
        num_splits,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_se_conv_sigmoid_mul_gelu_avgpool_flatten_dropout