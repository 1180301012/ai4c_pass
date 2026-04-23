import torch
import triton
import triton.language as tl


OUT_H = 64
OUT_W = 128
IN_W = 4


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def conv1x1_sigmoid_kernel(
    weight_ptr,
    x_ptr,
    gate_ptr,
    in_channels,
    out_channels,
    weight_stride_oc,
    weight_stride_ic,
    x_stride_n,
    x_stride_c,
    x_stride_w,
    gate_stride_n,
    gate_stride_c,
    gate_stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_cb = tl.program_id(1)

    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, IN_W)
    mask_c = offs_c < out_channels

    acc = tl.zeros((BLOCK_C, IN_W), dtype=tl.float32)

    for k_start in range(0, in_channels, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < in_channels

        w_ptrs = weight_ptr + offs_c[:, None] * weight_stride_oc + offs_k[None, :] * weight_stride_ic
        x_ptrs = x_ptr + pid_n * x_stride_n + offs_k[:, None] * x_stride_c + offs_w[None, :] * x_stride_w

        w = tl.load(w_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=mask_k[:, None], other=0.0)

        acc += tl.dot(w, x)

    sig = 1.0 / (1.0 + tl.exp(-acc))

    gate_ptrs = gate_ptr + pid_n * gate_stride_n + offs_c[:, None] * gate_stride_c + offs_w[None, :] * gate_stride_w
    tl.store(gate_ptrs, sig, mask=mask_c[:, None])


@triton.jit
def upsample_mul_kernel(
    gate_ptr,
    x_ptr,
    out_ptr,
    out_h,
    gate_stride_n,
    gate_stride_c,
    gate_stride_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hb = tl.program_id(2)

    offs_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, OUT_W)
    mask = (offs_h[:, None] < out_h)

    src = (offs_w.to(tl.float32) + 0.5) * (1.0 / 32.0) - 0.5
    src = tl.maximum(src, 0.0)
    x0 = tl.floor(src).to(tl.int32)
    at_end = x0 >= (IN_W - 1)
    x0_safe = tl.where(at_end, IN_W - 1, x0)
    x1 = tl.where(at_end, IN_W - 1, x0_safe + 1)
    lam = tl.where(at_end, 0.0, src - x0.to(tl.float32))

    g0 = tl.load(gate_ptr + pid_n * gate_stride_n + pid_c * gate_stride_c + x0_safe * gate_stride_w)
    g1 = tl.load(gate_ptr + pid_n * gate_stride_n + pid_c * gate_stride_c + x1 * gate_stride_w)
    gate = g0.to(tl.float32) + (g1.to(tl.float32) - g0.to(tl.float32)) * lam

    x_ptrs = (
        x_ptr
        + pid_n * x_stride_n
        + pid_c * x_stride_c
        + offs_h[:, None] * x_stride_h
        + offs_w[None, :] * x_stride_w
    )
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    out = x.to(tl.float32) * gate[None, :]

    out_ptrs = (
        out_ptr
        + pid_n * out_stride_n
        + pid_c * out_stride_c
        + offs_h[:, None] * out_stride_h
        + offs_w[None, :] * out_stride_w
    )
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_upsample_mul(in_0, in_1, in_2):
    n = in_1.shape[0]
    in_channels = in_1.shape[1]
    out_channels = in_0.shape[0]
    out_h = in_2.shape[2]

    gate = torch.empty((n, out_channels, IN_W), device=in_1.device, dtype=in_1.dtype)
    out = torch.empty_like(in_2)

    grid0 = (n, triton.cdiv(out_channels, 64))
    conv1x1_sigmoid_kernel[grid0](
        in_0,
        in_1,
        gate,
        in_channels,
        out_channels,
        in_0.stride(0),
        in_0.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        BLOCK_C=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )

    grid1 = (n, out_channels, triton.cdiv(out_h, 4))
    upsample_mul_kernel[grid1](
        gate,
        in_2,
        out,
        out_h,
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_H=4,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_conv_sigmoid_upsample_mul