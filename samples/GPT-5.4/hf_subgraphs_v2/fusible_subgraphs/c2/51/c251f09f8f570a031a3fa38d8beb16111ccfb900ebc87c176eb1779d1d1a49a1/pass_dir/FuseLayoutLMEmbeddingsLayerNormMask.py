import torch
import triton
import triton.language as tl


NEG_INF = -3.4028234663852886e+38
HIDDEN = 768


# Pattern matching function
# Use internal placeholders for the already-sliced / already-differenced index tensors
# so this matches all sequence lengths (11 / 64 / 128 / 256) and both bbox/mask input orders.
def pattern(
    input_ids,
    token_type_ids,
    pos_ids_sliced,
    ln_bias,
    ln_weight,
    hpos_w,
    pos_w,
    token_w,
    wpos_w,
    word_w,
    x_w,
    y_w,
    attention_mask,
    bbox0,
    bbox1,
    bbox2,
    bbox3,
    h_idx,
    w_idx,
):
    tmp_12 = attention_mask.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
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
    return (tmp_44, tmp_14)


# Argument extraction function
def replacement_args(
    input_ids,
    token_type_ids,
    pos_ids_sliced,
    ln_bias,
    ln_weight,
    hpos_w,
    pos_w,
    token_w,
    wpos_w,
    word_w,
    x_w,
    y_w,
    attention_mask,
    bbox0,
    bbox1,
    bbox2,
    bbox3,
    h_idx,
    w_idx,
):
    return (
        input_ids,
        token_type_ids,
        pos_ids_sliced,
        ln_bias,
        ln_weight,
        hpos_w,
        pos_w,
        token_w,
        wpos_w,
        word_w,
        x_w,
        y_w,
        attention_mask,
        bbox0,
        bbox1,
        bbox2,
        bbox3,
        h_idx,
        w_idx,
    )


@triton.jit
def _mask_to_bias_kernel(
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(mask_ptr + offs, mask=mask, other=1).to(tl.float32)
    y = (1.0 - x) * NEG_INF
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _layoutlm_embeddings_ln_kernel(
    input_ids_ptr,
    token_type_ids_ptr,
    pos_ids_ptr,
    bbox0_ptr,
    bbox1_ptr,
    bbox2_ptr,
    bbox3_ptr,
    h_idx_ptr,
    w_idx_ptr,
    word_w_ptr,
    pos_w_ptr,
    x_w_ptr,
    y_w_ptr,
    hpos_w_ptr,
    wpos_w_ptr,
    token_w_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_rows,
    seq_len,
    input_s0,
    input_s1,
    tok_s0,
    tok_s1,
    pos_s1,
    bbox0_s1,
    bbox1_s1,
    bbox2_s1,
    bbox3_s1,
    h_s1,
    w_s1,
    word_ws0,
    word_ws1,
    pos_ws0,
    pos_ws1,
    x_ws0,
    x_ws1,
    y_ws0,
    y_ws1,
    hpos_ws0,
    hpos_ws1,
    wpos_ws0,
    wpos_ws1,
    token_ws0,
    token_ws1,
    gamma_s0,
    beta_s0,
    out_s0,
    out_s1,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    batch_idx = row // seq_len
    seq_idx = row % seq_len

    word_idx = tl.load(input_ids_ptr + batch_idx * input_s0 + seq_idx * input_s1).to(tl.int32)
    tok_idx = tl.load(token_type_ids_ptr + batch_idx * tok_s0 + seq_idx * tok_s1).to(tl.int32)
    pos_idx = tl.load(pos_ids_ptr + seq_idx * pos_s1).to(tl.int32)
    bbox0_idx = tl.load(bbox0_ptr + seq_idx * bbox0_s1).to(tl.int32)
    bbox1_idx = tl.load(bbox1_ptr + seq_idx * bbox1_s1).to(tl.int32)
    bbox2_idx = tl.load(bbox2_ptr + seq_idx * bbox2_s1).to(tl.int32)
    bbox3_idx = tl.load(bbox3_ptr + seq_idx * bbox3_s1).to(tl.int32)
    hrow_idx = tl.load(h_idx_ptr + seq_idx * h_s1).to(tl.int32)
    wrow_idx = tl.load(w_idx_ptr + seq_idx * w_s1).to(tl.int32)

    sum_acc = tl.zeros((), dtype=tl.float32)
    sq_acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, HIDDEN, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        hmask = offs < HIDDEN

        v = tl.load(word_w_ptr + word_idx * word_ws0 + offs * word_ws1, mask=hmask, other=0.0)
        v = v + tl.load(pos_w_ptr + pos_idx * pos_ws0 + offs * pos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox0_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox1_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox2_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox3_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(hpos_w_ptr + hrow_idx * hpos_ws0 + offs * hpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(wpos_w_ptr + wrow_idx * wpos_ws0 + offs * wpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(token_w_ptr + tok_idx * token_ws0 + offs * token_ws1, mask=hmask, other=0.0)
        vf = v.to(tl.float32)
        sum_acc += tl.sum(vf, axis=0)
        sq_acc += tl.sum(vf * vf, axis=0)

    mean = sum_acc / HIDDEN
    var = sq_acc / HIDDEN - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + 1e-12)

    out_row_ptr = out_ptr + batch_idx * out_s0 + seq_idx * out_s1

    for start in range(0, HIDDEN, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        hmask = offs < HIDDEN

        v = tl.load(word_w_ptr + word_idx * word_ws0 + offs * word_ws1, mask=hmask, other=0.0)
        v = v + tl.load(pos_w_ptr + pos_idx * pos_ws0 + offs * pos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox0_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox1_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox2_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox3_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(hpos_w_ptr + hrow_idx * hpos_ws0 + offs * hpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(wpos_w_ptr + wrow_idx * wpos_ws0 + offs * wpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(token_w_ptr + tok_idx * token_ws0 + offs * token_ws1, mask=hmask, other=0.0)

        g = tl.load(gamma_ptr + offs * gamma_s0, mask=hmask, other=1.0).to(tl.float32)
        b = tl.load(beta_ptr + offs * beta_s0, mask=hmask, other=0.0).to(tl.float32)
        y = (v.to(tl.float32) - mean) * inv_std
        y = y * g + b
        tl.store(out_row_ptr + offs, y, mask=hmask)


@torch.fx.wrap
def layoutlm_embeddings_layernorm_mask_fused(
    input_ids,
    token_type_ids,
    pos_ids_sliced,
    ln_bias,
    ln_weight,
    hpos_w,
    pos_w,
    token_w,
    wpos_w,
    word_w,
    x_w,
    y_w,
    attention_mask,
    bbox0,
    bbox1,
    bbox2,
    bbox3,
    h_idx,
    w_idx,
):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    n_rows = batch_size * seq_len

    out = torch.empty((batch_size, seq_len, HIDDEN), device=word_w.device, dtype=word_w.dtype)
    mask_out = torch.empty_like(attention_mask, dtype=torch.float32)

    _layoutlm_embeddings_ln_kernel[(n_rows,)](
        input_ids,
        token_type_ids,
        pos_ids_sliced,
        bbox0,
        bbox1,
        bbox2,
        bbox3,
        h_idx,
        w_idx,
        word_w,
        pos_w,
        x_w,
        y_w,
        hpos_w,
        wpos_w,
        token_w,
        ln_weight,
        ln_bias,
        out,
        n_rows,
        seq_len,
        input_ids.stride(0),
        input_ids.stride(1),
        token_type_ids.stride(0),
        token_type_ids.stride(1),
        pos_ids_sliced.stride(1),
        bbox0.stride(1),
        bbox1.stride(1),
        bbox2.stride(1),
        bbox3.stride(1),
        h_idx.stride(1),
        w_idx.stride(1),
        word_w.stride(0),
        word_w.stride(1),
        pos_w.stride(0),
        pos_w.stride(1),
        x_w.stride(0),
        x_w.stride(1),
        y_w.stride(0),
        y_w.stride(1),
        hpos_w.stride(0),
        hpos_w.stride(1),
        wpos_w.stride(0),
        wpos_w.stride(1),
        token_w.stride(0),
        token_w.stride(1),
        ln_weight.stride(0),
        ln_bias.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_H=256,
        num_warps=8,
        num_stages=2,
    )

    n_mask = attention_mask.numel()
    _mask_to_bias_kernel[(triton.cdiv(n_mask, 1024),)](
        attention_mask,
        mask_out,
        n_mask,
        BLOCK_SIZE=1024,
    )

    return out, mask_out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return layoutlm_embeddings_layernorm_mask_fused