import torch
import triton
import triton.language as tl


def pattern(in_0, in_6):
    conv2d = torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_6):
    return (in_0, in_6)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
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
    H = 8
    W = 8
    K_OUT = 128
    K_TOTAL = 128 * 3 * 3

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m < (B * H * W)
    mask_n = n < K_OUT

    b = m // (H * W)
    hw = m % (H * W)
    p = hw // W
    q = hw % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_TOTAL, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K_TOTAL

        c = k // 9
        rs = k % 9
        r = rs // 3
        s = rs % 3

        h = p[:, None] + r[None, :] - 1
        w = q[:, None] + s[None, :] - 1

        x_ptrs = (
            x_ptr
            + b[:, None] * stride_xn
            + c[None, :] * stride_xc
            + h * stride_xh
            + w * stride_xw
        )
        x_mask = (
            mask_m[:, None]
            & mask_k[None, :]
            & (h >= 0)
            & (h < H)
            & (w >= 0)
            & (w < W)
        )
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = (
            w_ptr
            + n[None, :] * stride_wn
            + c[:, None] * stride_wc
            + r[:, None] * stride_wr
            + s[:, None] * stride_ws
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        wt = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, wt)

    out_ptrs = (
        out_ptr
        + b[:, None] * stride_on
        + n[None, :] * stride_oc
        + p[:, None] * stride_oh
        + q[:, None] * stride_ow
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def triton_conv2d_3x3_s1_p1(in_0, in_6):
    B = in_6.shape[0]
    out = torch.empty((B, 128, 8, 8), device=in_6.device, dtype=in_6.dtype)
    grid = lambda META: (
        triton.cdiv(B * 64, META["BLOCK_M"]),
        triton.cdiv(128, META["BLOCK_N"]),
    )
    _conv3x3s1p1_nchw_kernel[grid](
        in_6,
        in_0,
        out,
        B,
        in_6.stride(0),
        in_6.stride(1),
        in_6.stride(2),
        in_6.stride(3),
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return triton_conv2d_3x3_s1_p1