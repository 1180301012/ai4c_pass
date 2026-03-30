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
def fused_embed_add_ln_768_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_embed_ptr,
    pos_embed_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load indices for this token
    word_idx = tl.load(input_ids_ptr + pid)
    pos_idx = tl.load(position_ids_ptr + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load word embedding (cast idx to int64 for large vocab offsets)
    word_emb = tl.load(
        word_embed_ptr + word_idx.to(tl.int64) * hidden_dim + offsets,
        mask=mask, other=0.0
    )
    # Load position embedding
    pos_emb = tl.load(
        pos_embed_ptr + pos_idx.to(tl.int64) * hidden_dim + offsets,
        mask=mask, other=0.0
    )

    # Add embeddings in float32 for numerical stability
    x = word_emb.to(tl.float32) + pos_emb.to(tl.float32)

    # Layer norm: compute mean over valid elements
    x_masked = tl.where(mask, x, 0.0)
    x_sum = tl.sum(x_masked, axis=0)
    mean = x_sum / hidden_dim

    # Compute variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = tl.where(mask, (x - mean) * rstd, 0.0)

    # Apply LN weight and bias (loaded as float32)
    ln_w = tl.load(ln_weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = x_norm * ln_w + ln_b

    # Store float32 result
    out_offset = pid * hidden_dim
    tl.store(output_ptr + out_offset + offsets, output, mask=mask)


@torch.fx.wrap
def fused_embed_add_layernorm_768_1e05(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: input_ids [batch, seq_len]           int64
    # in_1: ln_bias [768]                         dtype
    # in_2: ln_weight [768]                       dtype
    # in_3: pos_embed_weight [pos_size, 768]      dtype
    # in_4: word_embed_weight [vocab_size, 768]   dtype
    # in_5: position_ids [batch, seq_len]         int64

    # Minimal wrapper — pass tensors directly, Triton handles conversion in registers
    N, S = in_0.shape[0], in_0.shape[1]
    total_tokens = N * S
    # Output in native dtype — Triton auto-converts float32 on store
    output = torch.empty((total_tokens, 768), dtype=in_1.dtype, device=in_5.device)

    fused_embed_add_ln_768_kernel[(total_tokens,)](
        in_0, in_5, in_4, in_3, in_2, in_1, output,
        hidden_dim=768, eps=1e-05, BLOCK_SIZE=1024, num_warps=4,
    )

    return output.view(N, S, 768)


def replacement_func():
    return fused_embed_add_layernorm_768_1e05