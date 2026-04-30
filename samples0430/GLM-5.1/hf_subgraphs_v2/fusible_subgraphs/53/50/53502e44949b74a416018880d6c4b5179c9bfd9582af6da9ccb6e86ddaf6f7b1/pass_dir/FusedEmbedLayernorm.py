import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5 = tmp_4 * 16.0
    tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7 = tmp_6.expand(1, -1)
    tmp_8 = tmp_7 + 2
    tmp_9 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    return (tmp_12,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_embed_ln_kernel(
    token_weight_ptr,
    pos_weight_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    input_ids_ptr,
    output_ptr,
    hidden_dim,
    pos_offset,
    scale,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load token index (first element of input_ids)
    token_idx = tl.load(input_ids_ptr)

    # Load token embedding row at token_idx
    token_embed = tl.load(
        token_weight_ptr + token_idx * hidden_dim + offsets, mask=mask, other=0.0
    )

    # Scale token embedding by 16.0
    scaled = token_embed * scale

    # Load position embedding at position offset (2)
    pos_embed = tl.load(
        pos_weight_ptr + pos_offset * hidden_dim + offsets, mask=mask, other=0.0
    )

    # Add token and position embeddings
    combined = scaled + pos_embed

    # Layer norm: compute mean
    masked_combined = tl.where(mask, combined, 0.0)
    mean = tl.sum(masked_combined) / hidden_dim

    # Compute variance
    diff = combined - mean
    masked_diff_sq = tl.where(mask, diff * diff, 0.0)
    var = tl.sum(masked_diff_sq) / hidden_dim

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Apply layer norm weight and bias
    ln_w = tl.load(ln_weight_ptr + offsets, mask=mask, other=0.0)
    ln_b = tl.load(ln_bias_ptr + offsets, mask=mask, other=0.0)
    result = ln_w * normalized + ln_b

    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_embed_ln(pos_weight, token_weight, ln_bias, ln_weight, input_ids):
    hidden_dim = token_weight.shape[1]
    dtype = token_weight.dtype
    dev = token_weight.device

    # Output shape matches the original: [1, 1, hidden_dim]
    output = torch.empty((1, 1, hidden_dim), dtype=dtype, device=dev)

    BLOCK_SIZE = 256  # Must be >= hidden_dim and a power of 2
    grid = (1,)

    fused_embed_ln_kernel[(grid,)](
        token_weight_ptr=token_weight,
        pos_weight_ptr=pos_weight,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        input_ids_ptr=input_ids,
        output_ptr=output,
        hidden_dim=hidden_dim,
        pos_offset=2,
        scale=16.0,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_embed_ln