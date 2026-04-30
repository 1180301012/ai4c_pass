import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_add_layernorm_kernel(
    input_ids_ptr,
    token_type_ids_ptr,
    position_ids_ptr,
    word_weight_ptr,
    token_weight_ptr,
    position_weight_ptr,
    gamma_ptr,
    beta_ptr,
    out_sum_ptr,
    out_ln_ptr,
    seq_len,
    hidden_size,
    stride_input_b,
    stride_input_s,
    stride_token_b,
    stride_token_s,
    stride_pos_s,
    stride_word_vocab,
    stride_word_hidden,
    stride_token_vocab,
    stride_token_hidden,
    stride_pos_vocab,
    stride_pos_hidden,
    stride_gamma,
    stride_beta,
    stride_out_sum_b,
    stride_out_sum_s,
    stride_out_sum_h,
    stride_out_ln_b,
    stride_out_ln_s,
    stride_out_ln_h,
    eps,
    STORE_SUM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < hidden_size

    word_idx = tl.load(input_ids_ptr + b * stride_input_b + s * stride_input_s)
    token_idx = tl.load(token_type_ids_ptr + b * stride_token_b + s * stride_token_s)
    pos_idx = tl.load(position_ids_ptr + s * stride_pos_s)

    word = tl.load(
        word_weight_ptr + word_idx * stride_word_vocab + offs_h * stride_word_hidden,
        mask=h_mask,
        other=0,
    )
    token = tl.load(
        token_weight_ptr + token_idx * stride_token_vocab + offs_h * stride_token_hidden,
        mask=h_mask,
        other=0,
    )
    pos = tl.load(
        position_weight_ptr + pos_idx * stride_pos_vocab + offs_h * stride_pos_hidden,
        mask=h_mask,
        other=0,
    )

    summed = word + token
    summed = summed + pos

    x = summed.to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_h * stride_gamma, mask=h_mask, other=1).to(tl.float32)
    beta = tl.load(beta_ptr + offs_h * stride_beta, mask=h_mask, other=0).to(tl.float32)
    y = x_centered * inv_std
    y = y * gamma + beta
    y = y.to(summed.dtype)

    if STORE_SUM:
        tl.store(
            out_sum_ptr + b * stride_out_sum_b + s * stride_out_sum_s + offs_h * stride_out_sum_h,
            summed,
            mask=h_mask,
        )

    tl.store(
        out_ln_ptr + b * stride_out_ln_b + s * stride_out_ln_s + offs_h * stride_out_ln_h,
        y,
        mask=h_mask,
    )


@torch.fx.wrap
def dispatch_embedding_add_layernorm(
    input_ids,
    word_weight,
    token_weight,
    position_weight,
    gamma,
    beta,
    token_type_ids,
    position_ids,
    route,
):
    batch, seq_len = input_ids.shape
    hidden_size = gamma.shape[0]

    out_ln = torch.empty((batch, seq_len, hidden_size), device=word_weight.device, dtype=word_weight.dtype)

    if route == "ret2":
        out_sum = torch.empty((batch, seq_len, hidden_size), device=word_weight.device, dtype=word_weight.dtype)
        store_sum = True
        out_sum_ptr = out_sum
    else:
        out_sum = None
        store_sum = False
        out_sum_ptr = out_ln

    if hidden_size <= 32:
        block_h = 32
        num_warps = 1
    elif hidden_size <= 64:
        block_h = 64
        num_warps = 2
    elif hidden_size <= 768:
        block_h = 1024
        num_warps = 4
    else:
        block_h = 1024
        num_warps = 8

    grid = (batch * seq_len,)
    _embedding_add_layernorm_kernel[grid](
        input_ids,
        token_type_ids,
        position_ids,
        word_weight,
        token_weight,
        position_weight,
        gamma,
        beta,
        out_sum_ptr,
        out_ln,
        seq_len,
        hidden_size,
        input_ids.stride(0),
        input_ids.stride(1),
        token_type_ids.stride(0),
        token_type_ids.stride(1),
        position_ids.stride(1),
        word_weight.stride(0),
        word_weight.stride(1),
        token_weight.stride(0),
        token_weight.stride(1),
        position_weight.stride(0),
        position_weight.stride(1),
        gamma.stride(0),
        beta.stride(0),
        out_sum_ptr.stride(0),
        out_sum_ptr.stride(1),
        out_sum_ptr.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        1e-12,
        STORE_SUM=store_sum,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=2,
    )

    if route == "ret2":
        return (out_sum, out_ln)
    return (out_ln,)