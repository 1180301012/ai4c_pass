import torch
import triton
import triton.language as tl


# Match the full returned subgraph exactly.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return (conv2d, tmp_8)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["B"],
)
@triton.jit
def _conv3x3s1p1_nchw_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    B,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wr,
    stride_ws,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Specialized for this graph family:
    # input  [B, 128, 8, 8]
    # weight [128, 128, 3, 3]
    # output [B, 128, 8, 8]
    C = 128
    H = 8
    W = 8
    K_OUT = 128
    K_TOTAL = 1152  # 128 * 3 * 3

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < (B * H * W)
    mask_n = n_offsets < K_OUT

    b = m_offsets // (H * W)
    hw = m_offsets % (H * W)
    p = hw // W
    q = hw % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_TOTAL

        c = k_offsets // 9
        rs = k_offsets % 9
        r = rs // 3
        s = rs % 3

        h_in = p[:, None] + r[None, :] - 1
        w_in = q[:, None] + s[None, :] - 1

        x_ptrs = (
            x_ptr
            + b[:, None] * stride_xn
            + c[None, :] * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        x_mask = (
            mask_m[:, None]
            & mask_k[None, :]
            & (h_in >= 0)
            & (h_in < H)
            & (w_in >= 0)
            & (w_in < W)
        )
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = (
            w_ptr
            + n_offsets[None, :] * stride_wn
            + c[:, None] * stride_wc
            + r[:, None] * stride_wr
            + s[:, None] * stride_ws
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    out_ptrs = (
        out_ptr
        + b[:, None] * stride_on
        + n_offsets[None, :] * stride_oc
        + p[:, None] * stride_oh
        + q[:, None] * stride_ow
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
    ],
    key=["B"],
)
@triton.jit
def _avgpool_bn_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    B,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    BLOCK: tl.constexpr,
):
    # Specialized for this graph family:
    # input [B, 512, 8, 8]
    # output [B, 512, 1, 1]
    C = 512
    H = 8
    W = 8
    HW = 64
    eps = 1e-5

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (B * C)

    b = offsets // C
    c = offsets % C

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h in range(8):
        for w in range(8):
            ptrs = (
                x_ptr
                + b * stride_xn
                + c * stride_xc
                + h * stride_xh
                + w * stride_xw
            )
            val = tl.load(ptrs, mask=mask, other=0.0)
            acc += val.to(tl.float32)

    avg = acc * (1.0 / HW)
    mean = tl.load(running_mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    y = (avg - mean) * tl.rsqrt(var + eps)
    y = y * gamma + beta
    y = tl.maximum(y, 0.0)

    out_ptrs = out_ptr + b * stride_on + c * stride_oc
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_ddrnet_spp_conv_pool_bn_relu(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Conv branch
    B = in_6.shape[0]
    conv_out = torch.empty((B, 128, 8, 8), device=in_6.device, dtype=in_6.dtype)
    conv_grid = lambda META: (
        triton.cdiv(B * 64, META["BLOCK_M"]),
        triton.cdiv(128, META["BLOCK_N"]),
    )
    _conv3x3s1p1_nchw_kernel[conv_grid](
        in_6,
        in_0,
        conv_out,
        B,
        in_6.stride(0),
        in_6.stride(1),
        in_6.stride(2),
        in_6.stride(3),
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
    )

    # AdaptiveAvgPool2d(1,1) + BatchNorm(inference) + ReLU branch
    pool_bn_relu_out = torch.empty((in_5.shape[0], 512, 1, 1), device=in_5.device, dtype=in_5.dtype)
    abr_grid = lambda META: (triton.cdiv(in_5.shape[0] * 512, META["BLOCK"]),)
    _avgpool_bn_relu_kernel[abr_grid](
        in_5,
        in_1,
        in_2,
        in_3,
        in_4,
        pool_bn_relu_out,
        in_5.shape[0],
        in_5.stride(0),
        in_5.stride(1),
        in_5.stride(2),
        in_5.stride(3),
        pool_bn_relu_out.stride(0),
        pool_bn_relu_out.stride(1),
    )

    return (conv_out, pool_bn_relu_out)


def replacement_func():
    return fused_ddrnet_spp_conv_pool_bn_relu