import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full unfold → permute → reshape → cat → cast graph
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Output layout  [35, 3, 384, 384]  (float16)
#   [  0 .. 24]  patches from in_2  (25 patches, nW=5, stride=288)
#   [ 25 .. 33]  patches from in_1  ( 9 patches, nW=3, stride=192)
#   [34       ]  in_0               ( 1 image)
#
# Grid: one program per output row (n, c, h)  →  35 * 3 * 384 = 40 320 programs
# Each program writes 384 consecutive float16 values (w dimension).
# BW=512 (power-of-2) with masking keeps address computation simple.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_unfold_cat_cast_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    # runtime strides (batch dim is always 1, so we skip stride-0)
    in0_sc, in0_sh,   # in_0 channel / height strides
    in1_sc, in1_sh,   # in_1 channel / height strides
    in2_sc, in2_sh,   # in_2 channel / height strides
    out_sn, out_sc, out_sh,   # output n / channel / height strides
    # compile-time constants
    N_P2: tl.constexpr,   # patches from in_2  = 25
    NW2:  tl.constexpr,   # patches per row    =  5
    N_P1: tl.constexpr,   # patches from in_1  =  9
    NW1:  tl.constexpr,   # patches per row    =  3
    SH2:  tl.constexpr,   # patch stride-H in in_2 = 288
    SW2:  tl.constexpr,   # patch stride-W in in_2 = 288
    SH1:  tl.constexpr,   # patch stride-H in in_1 = 192
    SW1:  tl.constexpr,   # patch stride-W in in_1 = 192
    KH:   tl.constexpr,   # kernel / patch height  = 384
    KW:   tl.constexpr,   # kernel / patch width   = 384
    C:    tl.constexpr,   # channels               =   3
    BW:   tl.constexpr,   # block width (power-of-2 >= KW) = 512
):
    # Decode which output row this program covers
    pid     = tl.program_id(0)
    h       = pid % KH
    tmp_pid = pid // KH
    c       = tmp_pid % C
    n       = tmp_pid // C

    w      = tl.arange(0, BW)
    w_mask = w < KW

    out_off = n * out_sn + c * out_sc + h * out_sh + w

    # ---- source: in_2 --------------------------------------------------------
    is_in2 = n < N_P2
    pi2    = tl.where(is_in2, n // NW2, 0)
    pj2    = tl.where(is_in2, n % NW2,  0)
    sh2    = pi2 * SH2 + h
    sw2    = pj2 * SW2
    off2   = c * in2_sc + sh2 * in2_sh + sw2 + w
    val2   = tl.load(in2_ptr + off2, mask=(w_mask & is_in2), other=0.0)

    # ---- source: in_1 --------------------------------------------------------
    is_in1 = (n >= N_P2) & (n < N_P2 + N_P1)
    n1     = tl.where(is_in1, n - N_P2, 0)
    pi1    = n1 // NW1
    pj1    = n1 % NW1
    sh1    = pi1 * SH1 + h
    sw1    = pj1 * SW1
    off1   = c * in1_sc + sh1 * in1_sh + sw1 + w
    val1   = tl.load(in1_ptr + off1, mask=(w_mask & is_in1), other=0.0)

    # ---- source: in_0 (last slice) -------------------------------------------
    is_in0 = n >= N_P2 + N_P1
    off0   = c * in0_sc + h * in0_sh + w
    val0   = tl.load(in0_ptr + off0, mask=(w_mask & is_in0), other=0.0)

    # Combine and cast to float16
    val = tl.where(is_in2, val2, tl.where(is_in1, val1, val0)).to(tl.float16)

    tl.store(out_ptr + out_off, val, mask=w_mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX treats it as an opaque node)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_unfold_cat_cast(in_0, in_1, in_2):
    # ---- fixed geometry (matches weight_meta.py) ----------------------------
    N_P2, NW2 = 25, 5          # in_2: 5×5 patches
    N_P1, NW1 = 9,  3          # in_1: 3×3 patches
    SH2, SW2  = 288, 288
    SH1, SW1  = 192, 192
    KH, KW, C = 384, 384, 3
    BW        = 512             # next power-of-2 >= KW=384
    N_TOTAL   = N_P2 + N_P1 + 1  # 35

    out = torch.empty((N_TOTAL, C, KH, KW), dtype=torch.float16, device=in_0.device)

    grid = (N_TOTAL * C * KH,)   # 35 * 3 * 384 = 40 320

    _fused_unfold_cat_cast_kernel[grid](
        in_0, in_1, in_2, out,
        # strides (channel & height only; batch=0 and w-stride=1 are implicit)
        in_0.stride(1), in_0.stride(2),
        in_1.stride(1), in_1.stride(2),
        in_2.stride(1), in_2.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        # constexpr args
        N_P2=N_P2, NW2=NW2,
        N_P1=N_P1, NW1=NW1,
        SH2=SH2, SW2=SW2,
        SH1=SH1, SW1=SW1,
        KH=KH, KW=KW, C=C,
        BW=BW,
        # launch config
        num_warps=8,
    )

    return (out,)


def replacement_func():
    return _fused_unfold_cat_cast