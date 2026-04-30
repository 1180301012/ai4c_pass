import torch
import triton
import triton.language as tl


@triton.jit
def _conv1x1_sigmoid_w4_kernel(
    weight_ptr,
    input_ptr,
    tmp_ptr,
    C_IN,
    C_OUT,
    IN_W,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    if c >= C_OUT:
        return

    offs_k = tl.arange(0, BLOCK_K)
    sum0 = tl.zeros([], dtype=tl.float32)
    sum1 = tl.zeros([], dtype=tl.float32)
    sum2 = tl.zeros([], dtype=tl.float32)
    sum3 = tl.zeros([], dtype=tl.float32)

    k = 0
    while k < C_IN:
        k_idx = k + offs_k
        mask = k_idx < C_IN

        w = tl.load(weight_ptr + c * C_IN + k_idx, mask=mask, other=0.0).to(tl.float32)
        x0 = tl.load(input_ptr + k_idx * IN_W + 0, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(input_ptr + k_idx * IN_W + 1, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(input_ptr + k_idx * IN_W + 2, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(input_ptr + k_idx * IN_W + 3, mask=mask, other=0.0).to(tl.float32)

        sum0 += tl.sum(w * x0, axis=0)
        sum1 += tl.sum(w * x1, axis=0)
        sum2 += tl.sum(w * x2, axis=0)
        sum3 += tl.sum(w * x3, axis=0)
        k += BLOCK_K

    y0 = tl.sigmoid(sum0)
    y1 = tl.sigmoid(sum1)
    y2 = tl.sigmoid(sum2)
    y3 = tl.sigmoid(sum3)

    base = c * IN_W
    tl.store(tmp_ptr + base + 0, y0)
    tl.store(tmp_ptr + base + 1, y1)
    tl.store(tmp_ptr + base + 2, y2)
    tl.store(tmp_ptr + base + 3, y3)


@triton.jit
def _upsample_mul_kernel(
    tmp_ptr,
    x_ptr,
    out_ptr,
    HW,
    OUT_W,
    BLOCK_HW: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    c = pid1
    offs = pid0 * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    ow = offs % OUT_W
    src = (ow.to(tl.float32) + 0.5) * (4.0 / OUT_W) - 0.5
    src = tl.maximum(src, 0.0)

    x0 = tl.floor(src).to(tl.int32)
    x1 = tl.minimum(x0 + 1, 3)
    w1 = src - tl.floor(src)
    w0 = 1.0 - w1

    base_tmp = c * 4
    v0 = tl.load(tmp_ptr + base_tmp + x0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(tmp_ptr + base_tmp + x1, mask=mask, other=0.0).to(tl.float32)
    gate = v0 * w0 + v1 * w1

    base_x = c * HW
    x = tl.load(x_ptr + base_x + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * gate
    tl.store(out_ptr + base_x + offs, out, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_interpolate_mul(in_0, in_1, in_2):
    c_out = in_0.shape[0]
    c_in = in_0.shape[1]
    in_w = in_1.shape[3]
    out_h = in_2.shape[2]
    out_w = in_2.shape[3]
    hw = out_h * out_w

    tmp = torch.empty((c_out, in_w), device=in_2.device, dtype=in_2.dtype)
    out = torch.empty_like(in_2)

    _conv1x1_sigmoid_w4_kernel[(c_out,)](
        in_0,
        in_1,
        tmp,
        c_in,
        c_out,
        in_w,
        BLOCK_K=128,
    )

    grid = (triton.cdiv(hw, 256), c_out)
    _upsample_mul_kernel[grid](
        tmp,
        in_2,
        out,
        hw,
        out_w,
        BLOCK_HW=256,
    )

    return out


def replacement_func():
    return fused_conv_sigmoid_interpolate_mul