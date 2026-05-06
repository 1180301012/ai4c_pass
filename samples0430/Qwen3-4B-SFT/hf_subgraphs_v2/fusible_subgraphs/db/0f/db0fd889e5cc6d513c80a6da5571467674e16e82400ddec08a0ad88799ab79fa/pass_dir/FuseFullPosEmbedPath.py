import torch
import triton
import triton.language as tl


def pattern(in_5, tmp_28, tmp_25):
    """
    Match the entire position-embedding processing path for in_5 and in_6.

    For in_5 ([1,236,32]):
      tmp_13 = in_5[:, 0, :]               -> CLS token  [1,32]
      tmp_14 = tmp_13[:, None]             -> CLS view    [1,1,32]
      tmp_15 = in_5[:, -10:, :]           -> det tokens  [1,10,32]
      tmp_16 = in_5[:, 1:-10, :]          -> main tokens [1,225,32]
      tmp_17 = tmp_16.transpose(1,2)       -> [1,225,32] (non-contiguous view)
      tmp_18 = tmp_17.view(1,32,15,15)
      tmp_19 = interpolate(tmp_18, (15,15)) -> bicubic NO-OP [1,32,15,15]
      tmp_20 = tmp_19.flatten(2)
      tmp_21 = tmp_20.transpose(1,2)       -> [1,225,32] (non-contiguous)
      tmp_22 = cat([tmp_14, tmp_21, tmp_15]) -> [1,242,32]

    For in_6 ([4,1,236,32]):
      tmp_25 = in_6[:,:,0,:]
      tmp_26 = tmp_25[:,:,None]
      tmp_27 = in_6[:,:,:-10,:]
      tmp_28 = in_6[:,:,1:-10,:]
      tmp_29 = tmp_28.transpose(2,3)        -> [4,32,15,15] (non-contiguous)
      tmp_30 = tmp_29.view(4,32,15,15)
      tmp_31 = interpolate(tmp_30,(15,15)) -> bicubic NO-OP [4,32,15,15]
      tmp_32 = tmp_31.flatten(2)
      tmp_33 = tmp_32.transpose(1,2)       -> [4,1,225,32] (non-contiguous)
      tmp_34 = tmp_33.contiguous()
      tmp_35 = tmp_34.view(4,1,225,32)

    Both bicubic interpolations are NO-OP (input == output size).
    The entire chain is just data rearrangement. We fuse everything into
    two Triton kernels:
      kernel 1: extract tmp_26 [1,1,32] and tmp_27 [1,10,32] from in_5/in_6
      kernel 2: produce tmp_35 [4,1,225,32] by rearranging/reshaping in_6
    """
    # --- in_5 path ---
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]   # [1,1,32]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_20 = tmp_19.flatten(2)
    tmp_21 = tmp_20.transpose(1, 2)
    tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)

    # --- in_6 path ---
    tmp_25 = tmp_25[(slice(None, None, None), slice(None, None, None), 0, slice(None, None, None))]
    tmp_26 = tmp_25[(slice(None, None, None), None)]   # [1,1,32]
    tmp_27 = tmp_27[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_28 = tmp_28[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)

    return (tmp_26, tmp_27, tmp_35)


def replacement_args(in_5, tmp_28, tmp_25):
    return (in_5, tmp_28, tmp_25)


# ---------------------------------------------------------------------------
# Kernel 1: extract tmp_26 (CLS) and tmp_27 (DET) from in_5 / in_6
# Output layout:
#   dst[0, 0, 0:C]   = in_5[0, 0, 0:C]   flat off 0..C-1
#   dst[0, 0, C:2C] = in_6[0,0, 225:235, 0:C]  flat off (225..235)*C
#   dst[0, 1, 0:C]   = in_6[0,0,     0: 10, 0:C] flat off (0..9) *C
# ---------------------------------------------------------------------------
@triton.jit
def extract_pos_emb_kernel(
    cls_in5_ptr,
    cls_in6_ptr,
    det_in5_ptr,
    det_in6_ptr,
    out_ptr,     # [1, 1, 32]
    out2_ptr,    # [1, 10, 32]
    C: tl.constexpr,      # 32
    TAIL_ROWS: tl.constexpr,    # 10
    TOTAL_SEQ: tl.constexpr,    # 236
):
    """
    One program per virtual channel slice.
    We split into two programs so we can conditionally assign rows:
      pid in [0, C):         CLS token  (only out_ptr)
      pid in [C, C+T*C):     DET token   (only out2_ptr)
    """
    pid = tl.program_id(0)
    C_TAIL = TAIL_ROWS * C

    if pid < C:
        ch = pid
        cls5 = tl.load(cls_in5_ptr + ch)
        cls6 = tl.load(cls_in6_ptr + ch)
        tl.store(out_ptr  + ch, cls5)
        tl.store(out2_ptr + ch, cls6)
    else:
        inner = pid - C
        ch = inner % C
        inner_i = inner // C
        if inner_i < TAIL_ROWS:
            raw_ch = inner_i * C + ch
            det5 = tl.load(det_in5_ptr + raw_ch)
            det6 = tl.load(det_in6_ptr + TAIL_ROWS * C + raw_ch)
            tl.store(out2_ptr + inner * C + ch, det5)
            tl.store(out2_ptr + (C_TAIL + inner) * C + ch, det6)


# ---------------------------------------------------------------------------
# Kernel 2: produce tmp_35 = reshape_cat_flatten(in_6)
#
# The bicubic NO-OP allows us to pass in_6 directly.
# Strategy:
#   out[i, 0, k, j] (i row of batch b, k spatial pos, j channel)  <- in_6[b,0,k,j]
# Which is: out[b*TOTAL_ROWS + k*TOTAL_C + j, j]  <- in_6[b*TOTAL_SEQ*CHANNELS + k*CHANNELS + j]
#
# Net: read in_6 sequentially; write output with transposition.
# in_6 strides: [TOTAL_SEQ*CHANNELS, TOTAL_SEQ*CHANNELS, CHANNELS, 1]
# out   strides: [TOTAL_ROWS*CHANNELS, CHANNELS, CHANNELS]  (contiguous [4,1,225,32])
# ---------------------------------------------------------------------------
@triton.jit
def extract_mid_seq_kernel(
    in6_ptr,
    out_ptr,
    HW:         tl.constexpr,   # H*W = 225
    SEQ:        tl.constexpr,   # 236
    CHANNELS:   tl.constexpr,   # 32
    TAIL_ROWS:  tl.constexpr,   # 10
    CLS_ROWS:   tl.constexpr,   # 1
):
    """
    One program per (batch, spatial, channel) triple.
    in_6[b, 0, k, j]  ->  out[b, 0, k, j]  (same values, different stride arrangement)
    """
    pid      = tl.program_id(0)    # in [0, TOTAL_B * [1,HW,CHANNELS] elements)
    total_c  = CHANNELS
    total_rows = HW          # 225
    total_b  = 4

    # Unpack (b, k, j) from flat pid
    j  = pid % total_c
    bk = pid // total_c
    k  = bk % total_rows
    b  = bk // total_rows

    in_offset  = b * SEQ * CHANNELS + k * CHANNELS + j
    out_offset = b * total_rows * CHANNELS * total_c + k * total_c + j

    val = tl.load(in6_ptr + in_offset)
    tl.store(out_ptr + out_offset, val)


@torch.fx.wrap
def fused_pos_emb_extractor(in_5, tmp_28, tmp_25):
    """
    in_5:     [1, 236, 32]  (position embeddings, batch 1)
    tmp_28:   [4, 1, 236, 32] (in_6 slice [b,:,1:236,:]
    tmp_25:   used to compute tmp_26

    Returns (tmp_26, tmp_27, tmp_35):
      tmp_26: [1, 1, 32]     cls token
      tmp_27: [1, 10, 32]    det tokens
      tmp_35: [4, 1, 225, 32] reshaped in_6 slice
    """
    C = 32
    TAIL_ROWS = 10
    TOTAL_SEQ = 236
    TOTAL_H   = 225   # H*W = 15*15

    # --- Kernel 1: extract CLS token (tmp_26) and DET tokens (tmp_27) ---
    out     = torch.empty((1, 1, C), dtype=in_5.dtype, device=in_5.device)
    out_det = torch.empty((1, TAIL_ROWS, C), dtype=in_5.dtype, device=in_5.device)

    extract_pos_emb_kernel[(C + TAIL_ROWS * C,)](
        in_5, in_5, # CLS for in_5 and in_6
        in_5,       # DET for in_5
        tmp_25,     # DET for in_6
        out,
        out_det,
        C=C,
        TAIL_ROWS=TAIL_ROWS,
        TOTAL_SEQ=TOTAL_SEQ,
    )

    # --- Kernel 2: reshape cat in_6 slice -> [4,1,225,32] (tmp_35) ---
    total_out = 4 * 1 * TOTAL_H * C      # 28 800
    out35     = torch.empty((4, 1, TOTAL_H, C), dtype=tmp_28.dtype, device=tmp_28.device)

    extract_mid_seq_kernel[(total_out,)](
        tmp_28, out35,
        HW=TOTAL_H,
        SEQ=TOTAL_SEQ,
        CHANNELS=C,
        TAIL_ROWS=TAIL_ROWS,
        CLS_ROWS=1,
    )

    return (out, out_det, out35)


def replacement_func():
    return fused_pos_emb_extractor