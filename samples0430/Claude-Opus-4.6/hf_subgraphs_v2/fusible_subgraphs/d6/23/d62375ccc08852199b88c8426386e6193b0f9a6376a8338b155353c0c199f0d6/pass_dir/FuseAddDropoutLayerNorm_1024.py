import torch
import triton
import triton.language as tl


def pattern(x, pos_embed, ln_bias, ln_weight):
    add_result = x + pos_embed
    dropout_result = torch.nn.functional.dropout(add_result, p=0.1, training=False)
    norm_result = torch.nn.functional.layer_norm(dropout_result, (1024,), ln_weight, ln_bias, 1e-05)
    return (dropout_result, norm_result)


def replacement_args(x, pos_embed, ln_bias, ln_weight):
    return (x, pos_embed, ln_weight, ln_bias)


@triton.jit
def _fused_add_layernorm_1024_kernel(
    x_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    out_sum_ptr,
    out_norm_ptr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = row_idx * BLOCK_SIZE
    x = tl.load(x_ptr + row_start + col_offsets)
    pos = tl.load(pos_ptr + row_start + col_offsets)

    # Add
    sum_val = x + pos

    # Store sum output (this is the "dropout" output since training=False)
    tl.store(out_sum_ptr + row_start + col_offsets, sum_val)

    # Layer norm computation in float32 for numerical accuracy
    sum_f32 = sum_val.to(tl.float32)
    mean = tl.sum(sum_f32, axis=0) / 1024.0
    centered = sum_f32 - mean
    var = tl.sum(centered * centered, axis=0) / 1024.0
    inv_std = tl.rsqrt(var + 1e-5)

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets)
    bias = tl.load(bias_ptr + col_offsets)

    # Normalize and apply affine transform
    normalized = centered * inv_std * weight.to(tl.float32) + bias.to(tl.float32)

    # Cast back to original dtype
    normalized = normalized.to(sum_val.dtype)

    # Store normalized output
    tl.store(out_norm_ptr + row_start + col_offsets, normalized)


@torch.fx.wrap
def fused_add_layernorm_1024(x, pos_embed, weight, bias):
    num_rows = x.numel() // 1024
    out_sum = torch.empty_like(x)
    out_norm = torch.empty_like(x)

    _fused_add_layernorm_1024_kernel[(num_rows,)](
        x,
        pos_embed,
        weight,
        bias,
        out_sum,
        out_norm,
        num_rows,
        BLOCK_SIZE=1024,
        num_warps=4,
    )

    return out_sum, out_norm


def replacement_func():
    return fused_add_layernorm_1024