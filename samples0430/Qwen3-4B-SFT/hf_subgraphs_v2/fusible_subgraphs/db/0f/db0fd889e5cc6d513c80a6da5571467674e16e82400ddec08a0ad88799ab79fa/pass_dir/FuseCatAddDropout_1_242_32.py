import torch
import triton
import triton.language as tl


def pattern(tmp_12, tmp_10, tmp_9, tmp_11, tmp_22):
    """
    Match:
      tmp_10 = in_3.expand(1,-1,-1)          # [1,1,32]
      tmp_11 = in_4.expand(1,-1,-1)          # [1,10,32]
      tmp_12 = cat([tmp_10, tmp_9, tmp_11])  # [1,225,32]
      tmp_22 = cat([tmp_14, tmp_21, tmp_15]) # [1,225,32]
      result  = tmp_12 + tmp_22
      output  = dropout(result, 0.1, False, False)  # training=False -> identity
    Fusing all three ops into one Triton kernel saves two intermediate buffers:
      - skip writing/delivering tmp_12
      - skip the dropout copy (pure no-op)
    """
    tmp_10_e = tmp_10.expand(1, -1, -1)
    tmp_11_e = tmp_11.expand(1, -1, -1)
    tmp_12_c = torch.cat((tmp_10_e, tmp_9, tmp_11_e), dim=1)
    tmp_22_c = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)
    result    = tmp_12_c + tmp_22_c
    output    = torch.nn.functional.dropout(result, 0.1, False, False)
    return output


def replacement_args(tmp_12, tmp_10, tmp_9, tmp_11, tmp_22):
    return (tmp_12, tmp_10, tmp_9, tmp_11, tmp_22)


@triton.jit
def cat_add_kernel(
    bias_ptr,                             # tmp_12:   [1, 242, 32]
    cls_ptr,                              # tmp_10 (or interpolated tmp_21):  [1, 1, 32]
    det_ptr,                              # tmp_11 (or last 10 of tmp_22):  [1, 10, 32]
    main_ptr,                             # tmp_9   (or interpolated tmp_21 sections): [1, 225, 32]
    bias_add_ptr,                         # tmp_22  [1, 242, 32]
    out_ptr,                              # output  [1, 242, 32]
    TOTAL_BIAS_ROWS: tl.constexpr,        # 242
    CLS_ROWS: tl.constexpr,              # 1
    DET_ROWS: tl.constexpr,              # 10
    MAIN_ROWS: tl.constexpr,             # 225
    CHANNELS: tl.constexpr,              # 32
):
    """
    One program per row of the output (one channel-slice).
    bias       [row, ch] -> tmp_12[0, row, ch]
    cls-source [0, ch]    -> (tmp_10[0,0,...] or tmp_21[0,0,...])[ch]
    det-source [row-MAIN,...] -> tmp_11[0, row-MAIN, ...][ch]
    main-source clamp -> tmp_9[row,...][ch]
    """
    row = tl.program_id(0)                       # in [0, TOTAL_BIAS_ROWS=242)
    ch  = tl.arange(0, CHANNELS)                 # [0..31]

    # --- load bias contribution: tmp_12[0, row, ch] ---
    bias = tl.load(bias_ptr + row * CHANNELS + ch)

    # --- load det contribution: tmp_11[0, row-MAIN, ch]  if row >= MAIN else 0 ---
    det_row = row - MAIN_ROWS
    valid_det = det_row >= 0                  # scalar bool
    det = tl.load(det_ptr + det_row * CHANNELS + ch, mask=valid_det, other=0.0)

    # --- load main contribution: tmp_9[row, ch] ---
    main = tl.load(main_ptr + row * CHANNELS + ch)

    # --- load bias-add contribution: tmp_22[0, row, ch] ---
    bias_add = tl.load(bias_add_ptr + row * CHANNELS + ch)

    # Select the right source for token_1 (middle section)
    tok_row2 = tl.where(row < CLS_ROWS, 0,
                        tl.where(row >= TOTAL_BIAS_ROWS, 0, row - CLS_ROWS))
    tok_valid2 = tl.where(row < CLS_ROWS, False,
                          tl.where(row >= TOTAL_BIAS_ROWS, False,
                                   tok_row2 >= 0 & (tok_row2 < MAIN_ROWS)))
    tok2 = tl.where(row < CLS_ROWS, 0.0,
                    tl.where(row >= TOTAL_BIAS_ROWS, 0.0,
                             tl.load(main_ptr + tok_row2 * CHANNELS + ch,
                                      mask=tok_valid2, other=0.0)))

    # Equivalent form for clamping + single masked load
    tok_row2_safe = tl.where(tok_valid2, tok_row2, 0)
    tok2 = tl.where(row < CLS_ROWS, 0.0,
                    tl.where(row >= TOTAL_BIAS_ROWS, 0.0,
                             tl.load(main_ptr + tok_row2_safe * CHANNELS + ch)))

    result = bias + tok2 + det + bias_add
    tl.store(out_ptr + row * CHANNELS + ch, result)


@torch.fx.wrap
def fused_cat_add_dropout(tmp_12, tmp_10, tmp_9, tmp_11, tmp_22):
    """
    tmp_12:  bias  [1, 242, 32]
    tmp_10:  cls   [1,   1, 32]  (= in_3 or interpolated equivalent)
    tmp_9:   main  [1, 225, 32]  (= tmp_21 after bicubic [skipped in caller])
    tmp_11:  det   [1,  10, 32]  (= in_4 or last 10 of tmp_22)
    tmp_22:  bias-add [1, 242, 32] (= in_6-proc after tmp_28 processing)
    output:  [1, 242, 32]
    """
    TOTAL_BIAS_ROWS = 242
    CLS_ROWS = 1
    DET_ROWS = 10
    MAIN_ROWS = 225
    CHANNELS = 32

    out = torch.empty_like(tmp_12)

    cat_add_kernel[(TOTAL_BIAS_ROWS,)](
        tmp_12, tmp_10, tmp_11, tmp_9, tmp_22, out,
        TOTAL_BIAS_ROWS=TOTAL_BIAS_ROWS,
        CLS_ROWS=CLS_ROWS,
        DET_ROWS=DET_ROWS,
        MAIN_ROWS=MAIN_ROWS,
        CHANNELS=CHANNELS,
    )
    return out


def replacement_func():
    return fused_cat_add_dropout