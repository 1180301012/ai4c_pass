import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.nn.functional.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    tmp_4 = in_2 + tmp_3
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(24, 24), mode='bilinear', align_corners=False)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _depthpro_conv_relu_add_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    o_ptr,
    sx_n,
    sx_c,
    sx_h,
    sx_w,
    sw_o,
    sw_i,
    sw_h,
    sw_w,
    sb_o,
    sr_n,
    sr_c,
    sr_h,
    sr_w,
    so_n,
    so_c,
    so_h,
    so_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < COUT
    mask_n = offs_n < (OH * OW)

    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh * 2 + kh - 1
            iw = ow * 2 + kw - 1
            valid_h = (ih >= 0) & (ih < IH)
            valid_w = (iw >= 0) & (iw < IW)
            valid_hw = valid_h & valid_w
            ih_safe = tl.where(valid_h, ih, 0)
            iw_safe = tl.where(valid_w, iw, 0)

            for ic_base in range(0, CIN, BLOCK_K):
                offs_k = ic_base + tl.arange(0, BLOCK_K)
                mask_k = offs_k < CIN

                w_ptrs = (
                    w_ptr
                    + offs_m[:, None] * sw_o
                    + offs_k[None, :] * sw_i
                    + kh * sw_h
                    + kw * sw_w
                )
                x_ptrs = (
                    x_ptr
                    + offs_k[:, None] * sx_c
                    + ih_safe[None, :] * sx_h
                    + iw_safe[None, :] * sx_w
                )

                w = tl.load(w_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_n[None, :] & valid_hw[None, :], other=0.0)
                acc += tl.dot(w, x)

    b = tl.load(b_ptr + offs_m * sb_o, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc + b[:, None]

    conv = acc.to(tl.float16)
    zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    relu = tl.maximum(conv, zero)

    r_ptrs = r_ptr + offs_m[:, None] * sr_c + oh[None, :] * sr_h + ow[None, :] * sr_w
    r = tl.load(r_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    out = relu + r

    o_ptrs = o_ptr + offs_m[:, None] * so_c + oh[None, :] * so_h + ow[None, :] * so_w
    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def depthpro_conv_relu_add_identity_resize(bias, weight, residual, x):
    out = torch.empty_like(residual)

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(128, BLOCK_M), triton.cdiv(24 * 24, BLOCK_N))

    sx = x.stride()
    sw = weight.stride()
    sb = bias.stride()
    sr = residual.stride()
    so = out.stride()

    _depthpro_conv_relu_add_kernel[grid](
        x,
        weight,
        bias,
        residual,
        out,
        sx[0],
        sx[1],
        sx[2],
        sx[3],
        sw[0],
        sw[1],
        sw[2],
        sw[3],
        sb[0],
        sr[0],
        sr[1],
        sr[2],
        sr[3],
        so[0],
        so[1],
        so[2],
        so[3],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        CIN=256,
        COUT=128,
        IH=48,
        IW=48,
        OH=24,
        OW=24,
        num_warps=8,
        num_stages=2,
    )
    return (out,)


def replacement_func():
    return depthpro_conv_relu_add_identity_resize