import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def emb_add_layernorm_h768_kernel(
    idx1_ptr,    # [N] int64 word token indices (flattened B*L)
    emb1_ptr,    # [V1, H] word embedding table
    idx2_ptr,    # [N] int64 position indices (flattened B*L)
    emb2_ptr,    # [V2, H] position embedding table
    weight_ptr,  # [H] LN weight
    bias_ptr,    # [H] LN bias
    out_ptr,     # [N, H] float32 output
    BLOCK_H: tl.constexpr,
):
    H = 768
    eps = 1e-05

    pid = tl.program_id(0)
    idx1 = tl.load(idx1_ptr + pid)
    idx2 = tl.load(idx2_ptr + pid)

    h_offs = tl.arange(0, BLOCK_H)
    mask = h_offs < H

    # Load embeddings (no max_norm check needed for inference with trained models)
    e1 = tl.load(emb1_ptr + idx1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
    e2 = tl.load(emb2_ptr + idx2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

    # Sum embeddings
    x = e1 + e2

    # Layer norm: mean and variance over valid elements
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid, axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    # Affine transform
    w = tl.load(weight_ptr + h_offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    tl.store(out_ptr + pid * H + h_offs, out, mask=mask)


@torch.fx.wrap
def fused_emb_add_layernorm_h768(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0: [B, L] int64  - word token indices
    in_1: [768]          - LN bias
    in_2: [768]          - LN weight
    in_3: [V2, 768]      - position embedding table
    in_4: [V1, 768]      - word embedding table
    in_5: [B, L] int64  - position indices
    """
    H = 768
    B, L = in_0.shape
    N = B * L

    orig_dtype = in_4.dtype
    device = in_4.device

    # Flatten indices to 1-D
    idx1 = in_0.reshape(-1).contiguous()
    idx2 = in_5.reshape(-1).contiguous()

    # Ensure embedding tables are contiguous
    emb1 = in_4.contiguous()
    emb2 = in_3.contiguous()

    # Cast LN params to float32 for stable computation
    weight = in_2.contiguous()
    bias = in_1.contiguous()

    # Allocate output in native dtype (Triton auto-casts float32 result on store)
    out_buf = torch.empty(N, H, dtype=orig_dtype, device=device)

    BLOCK_H = 1024  # next power-of-2 >= 768
    grid = (N,)

    emb_add_layernorm_h768_kernel[grid](
        idx1, emb1, idx2, emb2, weight, bias, out_buf,
        BLOCK_H=BLOCK_H,
    )

    # Reshape to [B, L, H] — no extra dtype conversion needed
    return out_buf.reshape(B, L, H)


def replacement_func():
    return fused_emb_add_layernorm_h768