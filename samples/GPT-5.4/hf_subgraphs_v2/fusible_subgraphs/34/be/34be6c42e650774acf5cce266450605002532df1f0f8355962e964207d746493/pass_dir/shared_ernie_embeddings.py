import torch
import triton
import triton.language as tl


@triton.jit
def _ernie_embeddings_layernorm_mask_kernel(
    input_ids_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    pos_weight_ptr,
    word_weight_ptr,
    out_ptr,
    mask_ptr,
    batch_size,
    seq_len,
    hidden_size,
    input_batch_stride,
    input_seq_stride,
    pos_row_stride,
    word_row_stride,
    out_batch_stride,
    out_seq_stride,
    mask_batch_stride,
    mask_seq_stride,
    eps,
    neg_large,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid - b * seq_len

    if b >= batch_size:
        return

    input_offset = b * input_batch_stride + s * input_seq_stride
    token_id = tl.load(input_ids_ptr + input_offset).to(tl.int32)
    pos_id = (s + 2).to(tl.int32)

    offs_h = tl.arange(0, BLOCK_H)
    hidden_mask = offs_h < hidden_size

    word_base = word_weight_ptr + token_id * word_row_stride
    pos_base = pos_weight_ptr + pos_id * pos_row_stride

    word_vals = tl.load(word_base + offs_h, mask=hidden_mask, other=0.0).to(tl.float32)
    pos_vals = tl.load(pos_base + offs_h, mask=hidden_mask, other=0.0).to(tl.float32)
    x = word_vals + pos_vals

    hidden_size_f = tl.cast(hidden_size, tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size_f
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size_f
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(ln_weight_ptr + offs_h, mask=hidden_mask, other=1.0).to(tl.float32)
    beta = tl.load(ln_bias_ptr + offs_h, mask=hidden_mask, other=0.0).to(tl.float32)
    y = centered * rstd
    y = y * gamma + beta

    out_offset = b * out_batch_stride + s * out_seq_stride + offs_h
    tl.store(out_ptr + out_offset, y, mask=hidden_mask)

    mask_offset = b * mask_batch_stride + s * mask_seq_stride
    mask_val = tl.where(token_id == 1, neg_large, 0.0)
    tl.store(mask_ptr + mask_offset, mask_val)


@torch.fx.wrap
def ernie_embeddings_dispatch(input_ids, ln_bias, ln_weight, pos_weight, word_weight, route):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_size = ln_weight.numel()

    out = torch.empty((batch_size, seq_len, hidden_size), dtype=word_weight.dtype, device=word_weight.device)
    mask = torch.empty((batch_size, 1, 1, seq_len), dtype=torch.float32, device=input_ids.device)

    if route == "h32":
        block_h = 32
        num_warps = 1
        num_stages = 2
    elif route == "h768":
        block_h = 1024
        num_warps = 8
        num_stages = 2
    else:
        if hidden_size <= 64:
            block_h = 64
            num_warps = 1
            num_stages = 2
        else:
            block_h = 1024
            num_warps = 8
            num_stages = 2

    grid = (batch_size * seq_len,)
    _ernie_embeddings_layernorm_mask_kernel[grid](
        input_ids,
        ln_weight,
        ln_bias,
        pos_weight,
        word_weight,
        out,
        mask,
        batch_size,
        seq_len,
        hidden_size,
        input_ids.stride(0),
        input_ids.stride(1),
        pos_weight.stride(0),
        word_weight.stride(0),
        out.stride(0),
        out.stride(1),
        mask.stride(0),
        mask.stride(3),
        1e-5,
        -3.4028234663852886e+38,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, mask