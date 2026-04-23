import torch
import triton
import triton.language as tl


def pattern(in_0, scale):
    tmp_0 = in_0 / scale
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)


def replacement_args(in_0, scale):
    return (in_0, scale)


# Specialized kernel for B3=8: processes whole row of 8 elements, BN columns at once
@triton.jit
def fused_mul_trans_b3_8_kernel(
    input_ptr, output_ptr,
    recip_scale,
    B0, B1, B2,
    s_in0, s_in1, s_in2,
    s_out0, s_out1, s_out2,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(B2, BN)

    bid = pid // n_tiles
    pn = pid % n_tiles

    b0 = bid // B1
    b1 = bid % B1

    on = pn * BN + tl.arange(0, BN)
    om = tl.arange(0, 8)

    mn = on < B2
    mm = om < 8

    ip = input_ptr + b0 * s_in0 + b1 * s_in1 + on[:, None] * s_in2 + om[None, :]
    msk = mn[:, None] & mm[None, :]
    d = tl.load(ip, mask=msk, other=0.0)
    d = d * recip_scale
    d = tl.trans(d)

    op = output_ptr + b0 * s_out0 + b1 * s_out1 + om[:, None] * s_out2 + on[None, :]
    mo = mm[:, None] & mn[None, :]
    tl.store(op, d, mask=mo)


# General kernel for arbitrary B3 (e.g., B3=64)
@triton.jit
def fused_mul_trans_general_kernel(
    input_ptr, output_ptr,
    recip_scale,
    B0, B1, B2, B3,
    s_in0, s_in1, s_in2, s_in3,
    s_out0, s_out1, s_out2, s_out3,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)

    n_tiles_m = tl.cdiv(B3, BM)
    n_tiles_n = tl.cdiv(B2, BN)
    n_inner = n_tiles_m * n_tiles_n

    bid = pid // n_inner
    iid = pid % n_inner
    pm = iid // n_tiles_n
    pn = iid % n_tiles_n

    b0 = bid // B1
    b1 = bid % B1

    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    mm = om < B3
    mn = on < B2

    ip = input_ptr + b0 * s_in0 + b1 * s_in1 + on[:, None] * s_in2 + om[None, :] * s_in3
    msk = mn[:, None] & mm[None, :]
    d = tl.load(ip, mask=msk, other=0.0)
    d = d * recip_scale
    d = tl.trans(d)

    op = output_ptr + b0 * s_out0 + b1 * s_out1 + om[:, None] * s_out2 + on[None, :] * s_out3
    mo = mm[:, None] & mn[None, :]
    tl.store(op, d, mask=mo)


@torch.fx.wrap
def fused_div_transpose(x, scale):
    B0, B1, B2, B3 = x.shape
    out = torch.empty(B0, B1, B3, B2, dtype=x.dtype, device=x.device)
    si = x.stride()
    so = out.stride()

    if B0 * B1 * B3 * B2 == 0:
        return out

    recip = 1.0 / scale

    if B3 == 8:
        # B3=8 specialized: whole row at once, BN columns per program
        BN = 16
        grid = (B0 * B1 * ((B2 + 15) >> 4),)
        fused_mul_trans_b3_8_kernel[grid](
            x, out, recip, B0, B1, B2,
            si[0], si[1], si[2],
            so[0], so[1], so[2],
            BN=BN, num_warps=2, num_stages=2,
        )
    elif B3 == 64:
        BN = 16
        grid = (B0 * B1 * ((B2 + 15) >> 4),)
        fused_mul_trans_general_kernel[grid](
            x, out, recip, B0, B1, B2, B3,
            si[0], si[1], si[2], si[3],
            so[0], so[1], so[2], so[3],
            BM=64, BN=BN, num_warps=4, num_stages=3,
        )
    else:
        BM = 8; BN = 8
        grid = (B0 * B1 * ((B3 + 7) >> 3) * ((B2 + 7) >> 3),)
        fused_mul_trans_general_kernel[grid](
            x, out, recip, B0, B1, B2, B3,
            si[0], si[1], si[2], si[3],
            so[0], so[1], so[2], so[3],
            BM=BM, BN=BN, num_warps=2, num_stages=2,
        )

    return out


def replacement_func():
    return fused_div_transpose