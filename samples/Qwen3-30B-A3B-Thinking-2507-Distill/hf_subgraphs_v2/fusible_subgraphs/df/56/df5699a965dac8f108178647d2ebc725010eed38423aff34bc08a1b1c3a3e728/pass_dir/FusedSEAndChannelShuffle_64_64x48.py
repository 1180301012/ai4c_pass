import torch
import triton
import triton.language as tl


@triton.jit
def _se_attn_64_64x48(
    in6_ptr, in1_ptr, bias_ptr, in5_ptr, out_ptr,
    B, C_in, C_out, HW,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C_out
    c_out = pid % C_out

    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < C_in
    v6 = tl.load(in6_ptr + b * C_in + k_idx, mask=k_mask, other=0.0).to(tl.float32)
    v1 = tl.load(in1_ptr + c_out * C_in + k_idx, mask=k_mask, other=0.0).to(tl.float32)
    cv = tl.sum(v6 * v1, axis=0)
    cv += tl.load(bias_ptr + c_out).to(tl.float32)
    att = tl.sigmoid(cv)

    hw_start = tl.program_id(1) * BLOCK_HW
    hw_idx = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < HW
    base = b * C_out * HW + c_out * HW
    x = tl.load(in5_ptr + base + hw_idx, mask=hw_mask)
    tl.store(out_ptr + base + hw_idx, x * att.to(x.dtype), mask=hw_mask)


@triton.jit
def _chan_shuffle_64_64x48(
    s1_ptr, s2_ptr, out0_ptr, out1_ptr,
    B, C, HW,
    C_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (2 * C_HALF)
    rem = pid % (2 * C_HALF)
    c = rem % C_HALF

    block_idx = tl.program_id(1)
    hw_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_idx < HW

    src_base = b * C * HW + c * HW
    val_s1 = tl.load(s1_ptr + src_base + hw_idx, mask=hw_mask)
    val_s2 = tl.load(s2_ptr + src_base + hw_idx, mask=hw_mask)

    out_even = b * C * HW + (2 * c) * HW + hw_idx
    out_odd = b * C * HW + (2 * c + 1) * HW + hw_idx
    tl.store(out0_ptr + out_even, val_s1, mask=hw_mask)
    tl.store(out0_ptr + out_odd, val_s2, mask=hw_mask)


@torch.fx.wrap
def _fused_b64_64x48(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    B, C_in, C_out = 64, 10, 40
    C2, C3 = 20, 40
    HW2, HW3 = 64 * 48, 32 * 24

    tmp_4 = torch.empty_like(in_5)
    out0 = torch.empty_like(in_2)
    out1 = torch.empty_like(in_2)
    out2 = torch.empty_like(in_3)
    out3 = torch.empty_like(in_3)

    BLOCK_HW, BLOCK_SZ = 256, 256

    nw = (HW3 + BLOCK_HW - 1) // BLOCK_HW
    _se_attn_64_64x48[(B * C_out, nw)](
        in_6, in_1, in_0, in_5, tmp_4,
        B, C_in, C_out, HW3,
        BLOCK_K=16, BLOCK_HW=BLOCK_HW,
    )

    nw2 = (HW2 + BLOCK_SZ - 1) // BLOCK_SZ
    _chan_shuffle_64_64x48[(B * C2, nw2)](
        in_2, in_4, out0, out1,
        B, C2, HW2, C_HALF=10, BLOCK_SIZE=BLOCK_SZ,
    )

    nw3 = (HW3 + BLOCK_SZ - 1) // BLOCK_SZ
    _chan_shuffle_64_64x48[(B * C3, nw3)](
        in_3, tmp_4, out2, out3,
        B, C3, HW3, C_HALF=20, BLOCK_SIZE=BLOCK_SZ,
    )

    return (out0, out2, out1, out3)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_7 = tmp_5.view(64, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(64, 40, 64, 48)
    tmp_11 = tmp_6.view(64, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(64, 80, 32, 24)
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_16, tmp_19, tmp_17, tmp_20)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return _fused_b64_64x48