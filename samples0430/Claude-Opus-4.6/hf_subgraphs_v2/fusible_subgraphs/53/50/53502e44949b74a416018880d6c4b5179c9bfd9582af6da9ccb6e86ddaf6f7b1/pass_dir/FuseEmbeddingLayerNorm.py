import torch
import triton
import triton.language as tl


def pattern(pos_weight, emb_weight, ln_bias, ln_weight, token_ids, pos_idx):
    tok_emb = torch.nn.functional.embedding(token_ids, emb_weight, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(pos_idx, pos_weight, None, None, 2.0, False, False)
    tmp = tok_emb * 16.0 + pos_emb
    return torch.nn.functional.layer_norm(tmp, (256,), ln_weight, ln_bias, 1e-05)


def replacement_args(pos_weight, emb_weight, ln_bias, ln_weight, token_ids, pos_idx):
    return (pos_weight, emb_weight, ln_bias, ln_weight, token_ids, pos_idx)


@triton.jit
def fused_embed_scale_add_layernorm_kernel(
    pos_weight_ptr, emb_weight_ptr, ln_bias_ptr, ln_weight_ptr,
    token_ids_ptr, pos_idx_ptr, out_ptr,
    HIDDEN: tl.constexpr,
):
    # Read indices
    token_id = tl.load(token_ids_ptr)
    pos_id = tl.load(pos_idx_ptr)

    offs = tl.arange(0, HIDDEN)

    # Load token embedding and scale
    tok_emb = tl.load(emb_weight_ptr + token_id * HIDDEN + offs).to(tl.float32)
    tok_emb = tok_emb * 16.0

    # Load position embedding
    pos_emb = tl.load(pos_weight_ptr + pos_id * HIDDEN + offs).to(tl.float32)

    # Add
    val = tok_emb + pos_emb

    # Layer norm
    mean = tl.sum(val, axis=0) / HIDDEN
    centered = val - mean
    var = tl.sum(centered * centered, axis=0) / HIDDEN
    normed = centered * tl.rsqrt(var + 1e-05)

    # Affine
    w = tl.load(ln_weight_ptr + offs).to(tl.float32)
    b = tl.load(ln_bias_ptr + offs).to(tl.float32)
    out = normed * w + b

    tl.store(out_ptr + offs, out)


@torch.fx.wrap
def fused_embed_scale_add_layernorm(pos_weight, emb_weight, ln_bias, ln_weight, token_ids, pos_idx):
    HIDDEN = 256
    out = torch.empty(1, 1, HIDDEN, dtype=pos_weight.dtype, device=pos_weight.device)
    fused_embed_scale_add_layernorm_kernel[(1,)](
        pos_weight, emb_weight, ln_bias, ln_weight,
        token_ids, pos_idx, out,
        HIDDEN=HIDDEN,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_embed_scale_add_layernorm