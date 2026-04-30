import torch
import triton
import triton.language as tl


def pattern(input_ids, word_table, token_type_ids, token_table, position_ids, pos_table, ln_weight, ln_bias, normalized_shape):
    word_emb = torch.nn.functional.embedding(input_ids, word_table, 0, None, 2.0, False, False)
    token_emb = torch.nn.functional.embedding(token_type_ids, token_table, None, None, 2.0, False, False)
    sum_wt = word_emb + token_emb
    pos_emb = torch.nn.functional.embedding(position_ids, pos_table, None, None, 2.0, False, False)
    sum_wt += pos_emb
    dropout_out = torch.nn.functional.dropout(sum_wt, 0.1, False, False)
    ln_out = torch.nn.functional.layer_norm(dropout_out, normalized_shape, ln_weight, ln_bias, 1e-12)
    return ln_out


def replacement_args(input_ids, word_table, token_type_ids, token_table, position_ids, pos_table, ln_weight, ln_bias, normalized_shape):
    return (input_ids, word_table, token_type_ids, token_table, position_ids, pos_table, ln_weight, ln_bias)


@triton.jit
def _fused_emb_add_ln_kernel_t2(
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_table_ptr, token_table_ptr, pos_table_ptr,
    ln_weight_ptr, ln_bias_ptr,
    ln_out_ptr,
    seq_len,
    hidden_dim,
    word_stride,
    token_stride,
    pos_stride,
    out_stride,
    BLOCK_D: tl.constexpr,
    POS_BROADCAST: tl.constexpr,
):
    row_idx = tl.program_id(0)
    seq_idx = row_idx % seq_len

    # Load indices
    word_idx = tl.load(input_ids_ptr + row_idx)
    token_idx = tl.load(token_type_ids_ptr + row_idx)
    if POS_BROADCAST:
        pos_idx = tl.load(position_ids_ptr + seq_idx)
    else:
        pos_idx = tl.load(position_ids_ptr + row_idx)

    # Column offsets
    cols = tl.arange(0, BLOCK_D)
    mask = cols < hidden_dim

    # Load embeddings
    word_emb = tl.load(word_table_ptr + word_idx * word_stride + cols, mask=mask, other=0.0)
    token_emb = tl.load(token_table_ptr + token_idx * token_stride + cols, mask=mask, other=0.0)
    pos_emb = tl.load(pos_table_ptr + pos_idx * pos_stride + cols, mask=mask, other=0.0)

    # Sum embeddings
    x = word_emb + token_emb + pos_emb

    # Layer norm in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / hidden_dim
    diff = tl.where(mask, x_f32 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / hidden_dim
    inv_std = 1.0 / tl.sqrt(var + 1e-12)
    x_normed = diff * inv_std

    # Apply LN weight and bias
    w = tl.load(ln_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln_result = x_normed * w + b

    # Store output
    out_base = row_idx * out_stride
    tl.store(ln_out_ptr + out_base + cols, ln_result.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_emb_add_ln_type2(input_ids, word_table, token_type_ids, token_table, position_ids, pos_table, ln_weight, ln_bias):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_dim = word_table.shape[1]
    total_rows = batch_size * seq_len

    pos_broadcast = position_ids.shape[0] != batch_size

    # Select BLOCK_D (must be power of 2 >= hidden_dim)
    if hidden_dim <= 32:
        BLOCK_D = 32
    elif hidden_dim <= 64:
        BLOCK_D = 64
    elif hidden_dim <= 128:
        BLOCK_D = 128
    elif hidden_dim <= 256:
        BLOCK_D = 256
    elif hidden_dim <= 512:
        BLOCK_D = 512
    elif hidden_dim <= 1024:
        BLOCK_D = 1024
    else:
        BLOCK_D = 2048

    # Select num_warps
    if BLOCK_D <= 64:
        num_warps = 2
    elif BLOCK_D <= 512:
        num_warps = 4
    else:
        num_warps = 8

    # Create output tensor
    ln_out = torch.empty(batch_size, seq_len, hidden_dim, dtype=word_table.dtype, device=word_table.device)

    # Launch kernel
    _fused_emb_add_ln_kernel_t2[(total_rows,)](
        input_ids, token_type_ids, position_ids,
        word_table, token_table, pos_table,
        ln_weight, ln_bias,
        ln_out,
        seq_len,
        hidden_dim,
        hidden_dim,  # word_stride
        hidden_dim,  # token_stride
        hidden_dim,  # pos_stride
        hidden_dim,  # out_stride
        BLOCK_D=BLOCK_D,
        POS_BROADCAST=pos_broadcast,
        num_warps=num_warps,
    )

    return ln_out


def replacement_func():
    return fused_emb_add_ln_type2