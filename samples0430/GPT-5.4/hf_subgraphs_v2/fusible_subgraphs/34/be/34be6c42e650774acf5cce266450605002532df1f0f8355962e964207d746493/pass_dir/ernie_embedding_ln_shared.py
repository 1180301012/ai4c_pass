import torch
import triton
import triton.language as tl


@triton.jit
def _ernie_embeddings_layernorm_and_mask_kernel(
    input_ids_ptr,
    word_weight_ptr,
    pos_weight_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_ptr,
    mask_ptr,
    batch_size,
    seq_len,
    hidden_size,
    input_stride_b,
    input_stride_s,
    word_stride_row,
    word_stride_col,
    pos_stride_row,
    pos_stride_col,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = batch_size * seq_len
    if pid >= total_rows:
        return

    b = pid // seq_len
    s = pid % seq_len

    input_offset = b * input_stride_b + s * input_stride_s
    token_id = tl.load(input_ids_ptr + input_offset)

    mask_value = tl.where(token_id == 1, -3.4028234663852886e+38, 0.0)
    tl.store(mask_ptr + pid, mask_value)

    pos_id = s + 2
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < hidden_size

    word_ptrs = word_weight_ptr + token_id * word_stride_row + cols * word_stride_col
    pos_ptrs = pos_weight_ptr + pos_id * pos_stride_row + cols * pos_stride_col

    word_vals = tl.load(word_ptrs, mask=col_mask, other=0.0)
    pos_vals = tl.load(pos_ptrs, mask=col_mask, other=0.0)

    x = tl.cast(word_vals, tl.float32) + tl.cast(pos_vals, tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(ln_weight_ptr + cols, mask=col_mask, other=1.0)
    beta = tl.load(ln_bias_ptr + cols, mask=col_mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(gamma, tl.float32) + tl.cast(beta, tl.float32)

    out_ptrs = out_ptr + b * out_stride_b + s * out_stride_s + cols * out_stride_h
    tl.store(out_ptrs, y, mask=col_mask)


def _pick_block_size_and_warps(hidden_size):
    if hidden_size <= 32:
        return 32, 1
    if hidden_size <= 64:
        return 64, 1
    if hidden_size <= 128:
        return 128, 2
    if hidden_size <= 256:
        return 256, 4
    if hidden_size <= 512:
        return 512, 4
    return 1024, 8


@torch.fx.wrap
def fused_ernie_embeddings_layernorm_and_mask(
    input_ids,
    ln_bias,
    ln_weight,
    pos_weight,
    word_weight,
):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_size = ln_weight.shape[0]

    out = torch.empty(
        (batch_size, seq_len, hidden_size),
        device=word_weight.device,
        dtype=word_weight.dtype,
    )
    mask = torch.empty(
        (batch_size, 1, 1, seq_len),
        device=input_ids.device,
        dtype=torch.float32,
    )

    block_size, num_warps = _pick_block_size_and_warps(hidden_size)
    grid = (batch_size * seq_len,)

    _ernie_embeddings_layernorm_and_mask_kernel[grid](
        input_ids,
        word_weight,
        pos_weight,
        ln_weight,
        ln_bias,
        out,
        mask,
        batch_size,
        seq_len,
        hidden_size,
        input_ids.stride(0),
        input_ids.stride(1),
        word_weight.stride(0),
        word_weight.stride(1),
        pos_weight.stride(0),
        pos_weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1e-5,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return out, mask


def replacement_func():
    return fused_ernie_embeddings_layernorm_and_mask