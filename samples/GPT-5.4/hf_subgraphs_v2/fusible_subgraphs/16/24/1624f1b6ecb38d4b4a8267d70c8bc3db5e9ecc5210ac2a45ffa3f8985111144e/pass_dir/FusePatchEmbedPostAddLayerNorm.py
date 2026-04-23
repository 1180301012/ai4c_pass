import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# Match the exact post-conv patch-embedding pipeline.
def pattern(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    tmp_7 = conv_out.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_embed
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), ln_weight, ln_bias, 1e-06)
    return (tmp_12, tmp_13)


# Preserve argument order for the replacement wrapper.
def replacement_args(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    return (conv_out, cls_token, pos_embed, ln_bias, ln_weight)


@triton.jit
def _vivit_patch_embed_post_ln_kernel(
    conv_ptr,
    cls_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    ln_out_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Fixed hidden size from the matched graph.
    HIDDEN = 768
    TOKENS = 980

    row = tl.program_id(0)
    row_offset = row * HIDDEN

    # Safe token index so the cls-token row does not issue an invalid load.
    safe_token = tl.where(row > 0, row - 1, 0)

    sum_acc = tl.zeros((), dtype=tl.float32)
    sqsum_acc = tl.zeros((), dtype=tl.float32)

    # Pass 1: compute mean / variance of tmp_12[row, :]
    for start in tl.static_range(0, HIDDEN, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < HIDDEN

        pos_vals = tl.load(pos_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        cls_vals = tl.load(cls_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        conv_vals = tl.load(conv_ptr + cols * TOKENS + safe_token, mask=mask, other=0.0).to(tl.float32)

        base_vals = tl.where(row == 0, cls_vals, conv_vals)
        x = base_vals + pos_vals

        sum_acc += tl.sum(x, axis=0)
        sqsum_acc += tl.sum(x * x, axis=0)

    mean = sum_acc / HIDDEN
    var = sqsum_acc / HIDDEN - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + eps)

    # Pass 2: write tmp_12 and layer-norm output.
    for start in tl.static_range(0, HIDDEN, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < HIDDEN

        pos_vals = tl.load(pos_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        cls_vals = tl.load(cls_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        conv_vals = tl.load(conv_ptr + cols * TOKENS + safe_token, mask=mask, other=0.0).to(tl.float32)

        base_vals = tl.where(row == 0, cls_vals, conv_vals)
        x = base_vals + pos_vals

        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * gamma + beta

        tl.store(out_ptr + row_offset + cols, x, mask=mask)
        tl.store(ln_out_ptr + row_offset + cols, y, mask=mask)


@torch.fx.wrap
def vivit_patch_embed_post_ln(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    conv_out = unwrap_tensor(conv_out)
    cls_token = unwrap_tensor(cls_token)
    pos_embed = unwrap_tensor(pos_embed)
    ln_bias = unwrap_tensor(ln_bias)
    ln_weight = unwrap_tensor(ln_weight)

    out = torch.empty_like(pos_embed)
    ln_out = torch.empty_like(pos_embed)

    rows = pos_embed.shape[1]

    _vivit_patch_embed_post_ln_kernel[(rows,)](
        conv_out,
        cls_token,
        pos_embed,
        ln_weight,
        ln_bias,
        out,
        ln_out,
        1e-6,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )

    return (out, ln_out)


def replacement_func():
    return vivit_patch_embed_post_ln