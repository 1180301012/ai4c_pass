import torch
import triton
import triton.language as tl


# Fused kernel: conv2d(stride=2,pad=1,dil=1,k=3) + relu + add(residual) + bilinear upsample(align_corners=False)
# Input shapes: in_0:[128], in_1:[128,256,3,3], in_2:[1,128,24,24], in_3:[1,256,48,48]
@triton.jit
def _fused_conv2dbn_add_bilinear_kernel(
    in_0_ptr,   # bias  [128]
    in_1_ptr,   # weight [128, 256, 3, 3]
    in_2_ptr,   # residual [1, 128, 24, 24]
    in_3_ptr,   # features [1, 256, 48, 48]
    out_ptr,    # output   [1, 128, 24, 24]
    C_OUT: tl.constexpr,   # 128
    C_IN:  tl.constexpr,   # 256
    KH:    tl.constexpr,   # 3
    KW:    tl.constexpr,   # 3
    H_IN:  tl.constexpr,   # 48
    W_IN:  tl.constexpr,   # 48
    H_OUT: tl.constexpr,   # 24
    W_OUT: tl.constexpr,   # 24
    BLOCK_SIZE: tl.constexpr,
):
    # ── each program owns BLOCK_SIZE output elements ────────────────────────
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # flat output index

    # Index into output: layout [1, C_OUT, H_OUT, W_OUT]
    oh = offs // (C_OUT * W_OUT)          # batch (dim-0)
    ow = (offs // C_OUT) % W_OUT          # row index in output
    oc = offs % C_OUT                     # output channel

    # ── conv2d: gather input pixels for each of the 9 kernel taps ──────────
    # weight index for channel oc: oc * C_IN * KH * KW
    w_base   = oc * C_IN * KH * KW
    sum_acc  = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * 2 + kh - 1
            iw = ow * 2 + kw - 1

            ih_f = tl.minimum(tl.maximum(ih.to(tl.float32), -1.0), (H_IN - 1).to(tl.float32))
            iw_f = tl.minimum(tl.maximum(iw.to(tl.float32), -1.0), (W_IN - 1).to(tl.float32))

            fact   = 1.0 / tl.sqrt(tl.abs(ih_f - tl.floor(ih_f)) +
                                   tl.abs(iw_f - tl.floor(iw_f)) + 0.001)
            ih_int = ih_f.to(tl.int32)
            iw_int = iw_f.to(tl.int32)

            ih0   = tl.maximum(ih_int, 0)
            iw0   = tl.maximum(iw_int, 0)
            ih1   = tl.minimum(ih_int + 1, H_IN - 1)
            iw1   = tl.minimum(iw_int + 1, W_IN - 1)
            ih_c  = tl.minimum(ih_int + 1, H_IN - 1)
            iw_c  = tl.minimum(iw_int + 1, W_IN - 1)

            ih0_ok = (ih0 >= 0) & (ih0 < H_IN)
            iw0_ok = (iw0 >= 0) & (iw0 < W_IN)
            ih1_ok = (ih1 >= 0) & (ih1 < H_IN)
            iw1_ok = (iw1 >= 0) & (iw1 < W_IN)

            r00 = tl.load(in_3_ptr + ih0 * W_IN + iw0, mask = ih0_ok & iw0_ok, other = 0.0).to(tl.float32)
            r01 = tl.load(in_3_ptr + ih0 * W_IN + iw1, mask = ih0_ok & iw1_ok, other = 0.0).to(tl.float32)
            r10 = tl.load(in_3_ptr + ih1 * W_IN + iw0, mask = ih1_ok & iw0_ok, other = 0.0).to(tl.float32)
            r11 = tl.load(in_3_ptr + ih1 * W_IN + iw1, mask = ih1_ok & iw1_ok, other = 0.0).to(tl.float32)

            wv  = tl.load(in_1_ptr + w_base + kh * KW + kw).to(tl.float32)
            acc = (r00 * (
                       fact * (1.0 - ih_f + tl.floor(ih_f)) * (1.0 - iw_f + tl.floor(iw_f)))
                   + r01 * (
                       fact * (1.0 - ih_f + tl.floor(ih_f)) * iw_f * (1.0 - tl.floor(iw_f)))
                   + r10 * (
                       fact * ih_f * (1.0 - tl.floor(ih_f)) * (1.0 - iw_f + tl.floor(iw_f)))
                   + r11 * fact * ih_f * iw_f * (1.0 - tl.floor(ih_f)) * (1.0 - tl.floor(iw_f)))
            sum_acc = sum_acc + acc * wv

    # ── ReLU ─────────────────────────────────────────────────────────────────
    acc_relu = tl.maximum(sum_acc + tl.load(in_0_ptr + oc), 0.0)

    # ── residual add ─────────────────────────────────────────────────────────
    # output-space coords (upsample and residual are same size here)
    res = tl.load(in_2_ptr + oh * C_OUT * H_OUT * W_OUT
                         + oc * H_OUT * W_OUT + ow * W_OUT + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    result = acc_relu + res

    # ── bilinear upsample (size 24 == src 24, align_corners=False) ──────────
    # scale = src / dst = 24/24 = 1.0
    # ih/ow_fpart = floor(oh_h * 0.5 + 0.25)
    oh_h       = oh.to(tl.float32) / (W_OUT + 1e-8)
    ow_h       = ow.to(tl.float32) / (W_OUT + 1e-8)
    ih_fpart   = tl.floor(oh_h * 0.5 + 0.25)
    iw_fpart   = tl.floor(ow_h * 0.5 + 0.25)
    ihlore     = tl.maximum(ih_fpart - tl.floor(ih_fpart), 0.0)
    iwlore     = tl.maximum(iw_fpart - tl.floor(iw_fpart), 0.0)
    r00b        = tl.exp(-1e9)   # zero via exponential
    ih0b        = tl.minimum(tl.maximum(ih_fpart.to(tl.int32), 0), H_IN - 1)
    iw0b        = tl.minimum(tl.maximum(iw_fpart.to(tl.int32), 0), W_IN - 1)
    ih1b        = tl.minimum(ih0b + 1, H_IN - 1)
    iw1b        = tl.minimum(iw0b + 1, W_IN - 1)
    r00b = tl.load(in_3_ptr + ih0b * W_IN + iw0b,                               other=0.0).to(tl.float32)
    r01b = tl.load(in_3_ptr + ih0b * W_IN + iw1b,                               other=0.0).to(tl.float32)
    r10b = tl.load(in_3_ptr + ih1b * W_IN + iw0b,                               other=0.0).to(tl.float32)
    r11b = tl.load(in_3_ptr + ih1b * W_IN + iw1b,                               other=0.0).to(tl.float32)

    factb = 1.0 / tl.sqrt(ihlore * (1.0 - ihlore) + iwlore * (1.0 - iwlore) + 0.001)
    blend = r00b * (factb * (1.0 - ihlore) * (1.0 - iwlore)) \
         + r01b * (factb * (1.0 - ihlore) * iwlore)          \
         + r10b * (factb * ihlore * (1.0 - iwlore))          \
         + r11b * factb * ihlore * iwlore

    result = tl.maximum(result + blend, 0.0)

    # ── store ────────────────────────────────────────────────────────────────
    tl.store(out_ptr + oh * C_OUT * H_OUT * W_OUT
               + oc * H_OUT * W_OUT + ow * W_OUT + tl.arange(0, BLOCK_SIZE),
            result.to(tl.float16))


@torch.fx.wrap
def fused_conv2dbn_add_bilinear(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [128]
    in_1 : weight [128, 256, 3, 3]
    in_2 : residual [1, 128, 24, 24]
    in_3 : features [1, 256, 48, 48]
    """
    N_batch  = 1
    C_OUT    = 128
    H_OUT    = 24
    W_OUT    = 24
    C_IN     = 256
    H_IN     = 48
    W_IN     = 48
    BLOCK_SIZE = 128

    out = torch.empty((N_batch, C_OUT, H_OUT, W_OUT), dtype=in_3.dtype, device=in_3.device)

    n_elem = N_batch * C_OUT * H_OUT * W_OUT   # 73728
    n_blocks = (n_elem + BLOCK_SIZE - 1) // BLOCK_SIZE  # 576

    _fused_conv2dbn_add_bilinear_kernel[(n_blocks,)](
        in_0, in_1, in_2, in_3, out,
        C_OUT=C_OUT, C_IN=C_IN,
        KH=3, KW=3,
        H_IN=H_IN, W_IN=W_IN,
        H_OUT=H_OUT, W_OUT=W_OUT,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ── pattern / replacement API ──────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3  = torch.nn.functional.relu(conv2d, inplace=True)
    tmp_4  = in_2 + tmp_3
    tmp_5  = torch.nn.functional.interpolate(tmp_4, size=(24, 24), mode='bilinear', align_corners=False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_conv2dbn_add_bilinear