import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p = 0.1, training = False)
    return (tmp_15,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "fuse_embed_add_layernorm")

@triton.jit
def fuse_embed_add_layernorm_kernel(
    input_embeds_ptr,
    embed_weight_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    cache_pos_ptr,
    out_ptr,
    seq_len,
    hidden_dim,
    stride_embed_num: tl.constexpr,
    stride_embed_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= seq_len:
        return

    # Load index from cache_position + 2
    pos = tl.load(cache_pos_ptr + row_idx).to(tl.int64) + 2

    # Pass 1: compute mean of (input_embeds + embedding)
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, hidden_dim, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < hidden_dim

        input_val = tl.load(input_embeds_ptr + row_idx * hidden_dim + cols, mask=col_mask, other=0.0).to(tl.float32)
        embed_val = tl.load(embed_weight_ptr + pos * stride_embed_dim + cols, mask=col_mask, other=0.0).to(tl.float32)

        val = input_val + embed_val
        _sum += val
        _sum_sq += val * val

    mean = tl.sum(_sum, axis=0) / hidden_dim
    var = tl.sum(_sum_sq, axis=0) / hidden_dim - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and store
    for start in range(0, hidden_dim, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < hidden_dim

        input_val = tl.load(input_embeds_ptr + row_idx * hidden_dim + cols, mask=col_mask, other=0.0).to(tl.float32)
        embed_val = tl.load(embed_weight_ptr + pos * stride_embed_dim + cols, mask=col_mask, other=0.0).to(tl.float32)

        w = tl.load(ln_weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        val = (input_val + embed_val - mean) * rstd * w + b

        tl.store(out_ptr + row_idx * hidden_dim + cols, val, mask=col_mask)

@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "fuse_mask":
        return _fuse_mask_impl(args[0])
    elif route == "fuse_embed_add_layernorm":
        return _fuse_embed_add_layernorm_impl(*args[:-1])
    else:
        raise RuntimeError(f"Unknown route: {route}")

def _fuse_mask_impl(in_5):
    raise RuntimeError("Not called from this pass")

def _fuse_embed_add_layernorm_impl(input_embeds, embed_weight, ln_bias, ln_weight, cache_position):
    batch_size = input_embeds.shape[0]
    seq_len = input_embeds.shape[1]
    hidden_dim = input_embeds.shape[2]

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    out = torch.empty_like(input_embeds)

    num_programs = batch_size * seq_len
    stride_embed_dim = embed_weight.shape[1]

    fuse_embed_add_layernorm_kernel[(num_programs,)](
        input_embeds_ptr=input_embeds,
        embed_weight_ptr=embed_weight,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        cache_pos_ptr=cache_position,
        out_ptr=out,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        stride_embed_num=embed_weight.shape[0],
        stride_embed_dim=stride_embed_dim,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)

def replacement_func():
    return shared_dispatch