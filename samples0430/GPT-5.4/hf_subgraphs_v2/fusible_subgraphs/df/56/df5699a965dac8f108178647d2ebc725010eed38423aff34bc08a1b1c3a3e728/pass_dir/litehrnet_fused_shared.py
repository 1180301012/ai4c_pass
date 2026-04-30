import torch
import triton
import triton.language as tl


@triton.jit
def _interleave_pair_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    b_s0,
    b_s1,
    b_s2,
    b_s3,
    o_s0,
    o_s1,
    o_s2,
    o_s3,
    width,
    hw,
    SRC_OFFSET: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < hw
    h = offs // width
    w = offs - h * width

    src_c = pid_c + SRC_OFFSET
    out_c0 = pid_c * 2
    out_c1 = out_c0 + 1

    a_idx = pid_b * a_s0 + src_c * a_s1 + h * a_s2 + w * a_s3
    b_idx = pid_b * b_s0 + src_c * b_s1 + h * b_s2 + w * b_s3
    o0_idx = pid_b * o_s0 + out_c0 * o_s1 + h * o_s2 + w * o_s3
    o1_idx = pid_b * o_s0 + out_c1 * o_s1 + h * o_s2 + w * o_s3

    a_val = tl.load(a_ptr + a_idx, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + b_idx, mask=mask, other=0.0)
    tl.store(out_ptr + o0_idx, a_val, mask=mask)
    tl.store(out_ptr + o1_idx, b_val, mask=mask)


@triton.jit
def _gated_interleave_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    y_s0,
    y_s1,
    y_s2,
    y_s3,
    z_s0,
    z_s1,
    w_s0,
    w_s1,
    o_s0,
    o_s1,
    o_s2,
    o_s3,
    width,
    hw,
    SRC_OFFSET: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    src_c = pid_c + SRC_OFFSET

    acc = tl.load(bias_ptr + src_c).to(tl.float32)
    for ci in range(10):
        z_val = tl.load(z_ptr + pid_b * z_s0 + ci * z_s1).to(tl.float32)
        w_val = tl.load(weight_ptr + src_c * w_s0 + ci * w_s1).to(tl.float32)
        acc += z_val * w_val
    gate = 1.0 / (1.0 + tl.exp(-acc))

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < hw
    h = offs // width
    w = offs - h * width

    out_c0 = pid_c * 2
    out_c1 = out_c0 + 1

    x_idx = pid_b * x_s0 + src_c * x_s1 + h * x_s2 + w * x_s3
    y_idx = pid_b * y_s0 + src_c * y_s1 + h * y_s2 + w * y_s3
    o0_idx = pid_b * o_s0 + out_c0 * o_s1 + h * o_s2 + w * o_s3
    o1_idx = pid_b * o_s0 + out_c1 * o_s1 + h * o_s2 + w * o_s3

    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + y_idx, mask=mask, other=0.0)
    gated_y = y_val.to(tl.float32) * gate

    tl.store(out_ptr + o0_idx, x_val, mask=mask)
    tl.store(out_ptr + o1_idx, gated_y, mask=mask)


@torch.fx.wrap
def fused_litehrnet_start68_end87(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    out_16 = torch.empty_like(in_2)
    out_17 = torch.empty_like(in_2)
    out_19 = torch.empty_like(in_3)
    out_20 = torch.empty_like(in_3)

    b0, _, h0, w0 = in_2.shape
    hw0 = h0 * w0
    block0 = 256
    grid0 = (b0, 10, triton.cdiv(hw0, block0))
    _interleave_pair_kernel[grid0](
        in_2,
        in_4,
        out_16,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_4.stride(0),
        in_4.stride(1),
        in_4.stride(2),
        in_4.stride(3),
        out_16.stride(0),
        out_16.stride(1),
        out_16.stride(2),
        out_16.stride(3),
        w0,
        hw0,
        SRC_OFFSET=0,
        BLOCK_HW=block0,
        num_warps=4,
        num_stages=2,
    )
    _interleave_pair_kernel[grid0](
        in_2,
        in_4,
        out_17,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_4.stride(0),
        in_4.stride(1),
        in_4.stride(2),
        in_4.stride(3),
        out_17.stride(0),
        out_17.stride(1),
        out_17.stride(2),
        out_17.stride(3),
        w0,
        hw0,
        SRC_OFFSET=10,
        BLOCK_HW=block0,
        num_warps=4,
        num_stages=2,
    )

    b1, _, h1, w1 = in_3.shape
    hw1 = h1 * w1
    block1 = 256
    grid1 = (b1, 20, triton.cdiv(hw1, block1))
    _gated_interleave_kernel[grid1](
        in_3,
        in_5,
        in_6,
        in_1,
        in_0,
        out_19,
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        in_5.stride(0),
        in_5.stride(1),
        in_5.stride(2),
        in_5.stride(3),
        in_6.stride(0),
        in_6.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        out_19.stride(0),
        out_19.stride(1),
        out_19.stride(2),
        out_19.stride(3),
        w1,
        hw1,
        SRC_OFFSET=0,
        BLOCK_HW=block1,
        num_warps=4,
        num_stages=2,
    )
    _gated_interleave_kernel[grid1](
        in_3,
        in_5,
        in_6,
        in_1,
        in_0,
        out_20,
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        in_5.stride(0),
        in_5.stride(1),
        in_5.stride(2),
        in_5.stride(3),
        in_6.stride(0),
        in_6.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        out_20.stride(0),
        out_20.stride(1),
        out_20.stride(2),
        out_20.stride(3),
        w1,
        hw1,
        SRC_OFFSET=20,
        BLOCK_HW=block1,
        num_warps=4,
        num_stages=2,
    )

    return out_16, out_19, out_17, out_20