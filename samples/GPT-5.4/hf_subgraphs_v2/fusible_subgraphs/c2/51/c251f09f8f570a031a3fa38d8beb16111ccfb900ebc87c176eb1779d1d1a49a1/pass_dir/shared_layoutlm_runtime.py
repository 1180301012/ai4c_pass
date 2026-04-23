import torch
import triton
import triton.language as tl


NEG_INF = -3.4028234663852886e+38
HIDDEN = 768


@triton.jit
def _mask_to_bias_kernel(
    mask_ptr,
    out_ptr,
    n_elements,
    NEG_INF_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements
    x = tl.load(mask_ptr + offs, mask=m, other=1).to(tl.float32)
    y = (1.0 - x) * NEG_INF_CONST
    tl.store(out_ptr + offs, y, mask=m)


@triton.jit
def _emb_ln_kernel(
    input_ids_ptr,
    pos_ids_ptr,
    bbox0_ptr,
    bbox1_ptr,
    bbox2_ptr,
    bbox3_ptr,
    h_idx_ptr,
    w_idx_ptr,
    token_type_ids_ptr,
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
    seq_len,
    input_s0,
    input_s1,
    pos_s1,
    bbox0_s1,
    bbox1_s1,
    bbox2_s1,
    bbox3_s1,
    h_s1,
    w_s1,
    tok_s0,
    tok_s1,
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
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len

    word_idx = tl.load(input_ids_ptr + b * input_s0 + s * input_s1).to(tl.int32)
    pos_idx = tl.load(pos_ids_ptr + s * pos_s1).to(tl.int32)
    bbox0_idx = tl.load(bbox0_ptr + s * bbox0_s1).to(tl.int32)
    bbox1_idx = tl.load(bbox1_ptr + s * bbox1_s1).to(tl.int32)
    bbox2_idx = tl.load(bbox2_ptr + s * bbox2_s1).to(tl.int32)
    bbox3_idx = tl.load(bbox3_ptr + s * bbox3_s1).to(tl.int32)
    h_idx = tl.load(h_idx_ptr + s * h_s1).to(tl.int32)
    w_idx = tl.load(w_idx_ptr + s * w_s1).to(tl.int32)
    tok_idx = tl.load(token_type_ids_ptr + b * tok_s0 + s * tok_s1).to(tl.int32)

    sum_acc = tl.zeros((), dtype=tl.float32)
    sq_acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, HIDDEN_SIZE, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        hmask = offs < HIDDEN_SIZE
        v = tl.load(word_w_ptr + word_idx * word_ws0 + offs * word_ws1, mask=hmask, other=0.0)
        v = v + tl.load(pos_w_ptr + pos_idx * pos_ws0 + offs * pos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox0_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox1_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox2_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox3_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(hpos_w_ptr + h_idx * hpos_ws0 + offs * hpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(wpos_w_ptr + w_idx * wpos_ws0 + offs * wpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(token_w_ptr + tok_idx * token_ws0 + offs * token_ws1, mask=hmask, other=0.0)
        vf = v.to(tl.float32)
        sum_acc += tl.sum(vf, axis=0)
        sq_acc += tl.sum(vf * vf, axis=0)

    mean = sum_acc / HIDDEN_SIZE
    var = sq_acc / HIDDEN_SIZE - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + 1e-12)
    out_row = out_ptr + b * out_s0 + s * out_s1

    for start in range(0, HIDDEN_SIZE, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        hmask = offs < HIDDEN_SIZE
        v = tl.load(word_w_ptr + word_idx * word_ws0 + offs * word_ws1, mask=hmask, other=0.0)
        v = v + tl.load(pos_w_ptr + pos_idx * pos_ws0 + offs * pos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox0_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox1_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(x_w_ptr + bbox2_idx * x_ws0 + offs * x_ws1, mask=hmask, other=0.0)
        v = v + tl.load(y_w_ptr + bbox3_idx * y_ws0 + offs * y_ws1, mask=hmask, other=0.0)
        v = v + tl.load(hpos_w_ptr + h_idx * hpos_ws0 + offs * hpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(wpos_w_ptr + w_idx * wpos_ws0 + offs * wpos_ws1, mask=hmask, other=0.0)
        v = v + tl.load(token_w_ptr + tok_idx * token_ws0 + offs * token_ws1, mask=hmask, other=0.0)
        g = tl.load(gamma_ptr + offs * gamma_s0, mask=hmask, other=1.0).to(tl.float32)
        bt = tl.load(beta_ptr + offs * beta_s0, mask=hmask, other=0.0).to(tl.float32)
        y = (v.to(tl.float32) - mean) * inv_std
        y = y * g + bt
        tl.store(out_row + offs, y, mask=hmask)


@torch.fx.wrap
def shared_layoutlm_dispatch(*args):
    route = args[-1]

    if route == "mask":
        attention_mask = args[0]
        out = torch.empty_like(attention_mask, dtype=torch.float32)
        n = attention_mask.numel()
        _mask_to_bias_kernel[(triton.cdiv(n, 1024),)](
            attention_mask,
            out,
            n,
            NEG_INF_CONST=NEG_INF,
            BLOCK_SIZE=1024,
        )
        return out

    if route == "emb_ln":
        (
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
            _route,
        ) = args
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        out = torch.empty((bsz, seq, HIDDEN), device=word_w.device, dtype=word_w.dtype)
        _emb_ln_kernel[(bsz * seq,)](
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
            out,
            seq,
            input_ids.stride(0),
            input_ids.stride(1),
            pos_ids_sliced.stride(1),
            bbox0.stride(1),
            bbox1.stride(1),
            bbox2.stride(1),
            bbox3.stride(1),
            h_idx.stride(1),
            w_idx.stride(1),
            token_type_ids.stride(0),
            token_type_ids.stride(1),
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
            HIDDEN_SIZE=HIDDEN,
            BLOCK_H=256,
            num_warps=8,
            num_stages=2,
        )
        return out

    return args[0]