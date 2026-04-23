import torch
import triton
import triton.language as tl
from pass_dir.shared_layoutlm_runtime import shared_layoutlm_dispatch


# Match the full 8-way embedding sum + layer_norm + inference dropout(identity)
def pattern(
    input_ids,
    pos_ids_sliced,
    bbox0,
    bbox1,
    bbox2,
    bbox3,
    h_idx,
    w_idx,
    token_type_ids,
    ln_bias,
    ln_weight,
    hpos_w,
    pos_w,
    token_w,
    wpos_w,
    word_w,
    x_w,
    y_w,
):
    tmp_16 = torch.nn.functional.embedding(input_ids, word_w, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(pos_ids_sliced, pos_w, None, None, 2.0, False, False)
    tmp_19 = torch.nn.functional.embedding(bbox0, x_w, None, None, 2.0, False, False)
    tmp_21 = torch.nn.functional.embedding(bbox1, y_w, None, None, 2.0, False, False)
    tmp_23 = torch.nn.functional.embedding(bbox2, x_w, None, None, 2.0, False, False)
    tmp_25 = torch.nn.functional.embedding(bbox3, y_w, None, None, 2.0, False, False)
    tmp_29 = torch.nn.functional.embedding(h_idx, hpos_w, None, None, 2.0, False, False)
    tmp_33 = torch.nn.functional.embedding(w_idx, wpos_w, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(token_type_ids, token_w, None, None, 2.0, False, False)
    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34
    tmp_43 = torch.nn.functional.layer_norm(tmp_42, (768,), ln_weight, ln_bias, 1e-12)
    tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
    return tmp_44


def replacement_args(
    input_ids,
    pos_ids_sliced,
    bbox0,
    bbox1,
    bbox2,
    bbox3,
    h_idx,
    w_idx,
    token_type_ids,
    ln_bias,
    ln_weight,
    hpos_w,
    pos_w,
    token_w,
    wpos_w,
    word_w,
    x_w,
    y_w,
):
    return (
        input_ids,
        pos_ids_sliced,
        bbox0,
        bbox1,
        bbox2,
        bbox3,
        h_idx,
        w_idx,
        token_type_ids,
        word_w,
        pos_w,
        x_w,
        y_w,
        hpos_w,
        wpos_w,
        token_w,
        ln_weight,
        ln_bias,
        "emb_ln",
    )


def replacement_func():
    return shared_layoutlm_dispatch