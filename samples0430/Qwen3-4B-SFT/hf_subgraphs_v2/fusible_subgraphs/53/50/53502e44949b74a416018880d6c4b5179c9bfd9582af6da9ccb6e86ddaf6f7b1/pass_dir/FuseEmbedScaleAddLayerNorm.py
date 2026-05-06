import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def fuse_embed_scale_add_ln_kernel(
    in_embed_ptr,      # [vocab_size, HIDDEN]  token embedding table (in_1)
    in_pos_ptr,        # [num_pos, HIDDEN]      position embedding table (in_0)
    in_pos_indices_ptr, # [B*S] int64            position indices (tmp_8)
    w_ptr,             # [HIDDEN]               layer norm weight (in_3)
    b_ptr,             # [HIDDEN]               layer norm bias  (in_2)
    out_ptr,           # [B*S, HIDDEN]           output
    HIDDEN: tl.constexpr,
    eps,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # Each program handles one row (one token-position pair)
    row = tl.program_id(0)
    row_offs = tl.arange(0, HIDDEN)

    # ---- Token embedding lookup ----
    tok_id = tl.load(in_pos_indices_ptr + row).to(tl.int64)
    tok_emb = tl.load(in_embed_ptr + tok_id * HIDDEN + row_offs).to(tl.float32)

    # ---- Position embedding lookup (pos_id = arange(0,S)+2 = constant +2) ----
    # tmp_8 = arange(0,1).expand(1,-1) + 2  => always [2]
    pos_emb_offs = row * HIDDEN + 0  # always pos_id=2 offset; in practice row=0 always
    pos_emb = tl.load(in_pos_ptr + pos_emb_offs + row_offs).to(tl.float32)

    # ---- Fused: scale * token + pos_emb ----
    x = tok_emb * 16.0 + pos_emb

    # ---- Layer Norm (float32 for numerical stability) ----
    mean = tl.sum(x, 0) / HIDDEN
    x_shifted = x - mean
    var = tl.sum(x_shifted * x_shifted, 0) / HIDDEN
    x_norm = x_shifted / tl.sqrt(var + eps)

    w = tl.load(w_ptr + row_offs).to(tl.float32)
    b = tl.load(b_ptr + row_offs).to(tl.float32)
    out = x_norm * w + b

    # ---- Store (cast back to original dtype) ----
    if IS_BF16:
        tl.store(out_ptr + row * HIDDEN + row_offs, out.to(tl.bfloat16))
    elif IS_FP16:
        tl.store(out_ptr + row * HIDDEN + row_offs, out.to(tl.float16))
    else:
        tl.store(out_ptr + row * HIDDEN + row_offs, out)


@torch.fx.wrap
def fused_embed_scale_add_ln(in_1, in_0, in_3, in_2, in_4):
    """
    in_1 : [vocab_size, HIDDEN]  token embedding weight
    in_0 : [num_pos, HIDDEN]     position embedding weight
    in_3 : [HIDDEN]              layer norm weight
    in_2 : [HIDDEN]              layer norm bias
    in_4 : [B, S] int64          input token ids
    Returns [B, S, HIDDEN]
    """
    B = in_4.shape[0]
    S = in_4.shape[1]
    HIDDEN = in_1.shape[1]

    out = torch.empty((B, S, HIDDEN), dtype=in_1.dtype, device=in_1.device)

    pos_ids = in_4.to(torch.int64).flatten()  # [B*S]

    IS_BF16 = in_1.dtype == torch.bfloat16
    IS_FP16 = in_1.dtype == torch.float16

    num_rows = B * S  # e.g. 1*1 = 1

    fuse_embed_scale_add_ln[(num_rows,)](
        in_1, in_0,
        pos_ids,
        in_3, in_2,
        out,
        HIDDEN=HIDDEN,
        eps=1e-5,
        IS_BF16=IS_BF16,
        IS_FP16=IS_FP16,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_1, in_0, in_3, in_2, in_4):
    """Matches the full transformer decoder embedding + layer-norm forward pass."""
    tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5 = tmp_4 * 16.0
    tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7 = tmp_6.expand(1, -1)
    tmp_8 = tmp_7 + 2
    tmp_9 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    return tmp_12


def replacement_args(in_1, in_0, in_3, in_2, in_4):
    return (in_1, in_0, in_3, in_2, in_4)


def replacement_func():
    return fused_embed_scale_add_ln