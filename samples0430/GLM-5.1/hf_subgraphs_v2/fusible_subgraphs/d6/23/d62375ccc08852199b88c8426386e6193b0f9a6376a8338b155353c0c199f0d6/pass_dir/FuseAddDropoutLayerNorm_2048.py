import torch
import triton
import triton.language as tl


def pattern(in_0, tmp_11, in_3, in_2):
    tmp_12 = in_0 + tmp_11
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2048,), in_3, in_2, 1e-05)
    return (tmp_13, tmp_14)


def replacement_args(in_0, tmp_11, in_3, in_2):
    return (in_0, tmp_11, in_3, in_2)


@triton.jit
def fused_add_layernorm_kernel_2048(
    x_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_norm_ptr,
    n_rows,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row_idx * BLOCK_SIZE

    # Load in native dtype
    x = tl.load(x_ptr + row_start + col_offsets)
    pos = tl.load(pos_ptr + row_start + col_offsets)

    # Add in native dtype (dropout with training=False is identity, so skip it)
    add_result = x + pos

    # Store add result (= tmp_13 since dropout is identity)
    tl.store(out_add_ptr + row_start + col_offsets, add_result)

    # Layer norm: upcast for numerical stability
    add_result_f32 = add_result.to(tl.float32)
    mean = tl.sum(add_result_f32, axis=0) / BLOCK_SIZE
    diff_f32 = add_result_f32 - mean
    variance = tl.sum(diff_f32 * diff_f32, axis=0) / BLOCK_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized_f32 = diff_f32 * rstd

    # Apply weight and bias
    weight = tl.load(weight_ptr + col_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets).to(tl.float32)
    norm_result_f32 = normalized_f32 * weight + bias

    # Store norm result (Triton will downcast if output is lower precision)
    tl.store(out_norm_ptr + row_start + col_offsets, norm_result_f32)


@torch.fx.wrap
def fused_add_layernorm_2048(x, pos_embed, weight, bias):
    n_rows = x.shape[0] * x.shape[1]

    out_add = torch.empty_like(x)
    out_norm = torch.empty_like(x)

    grid = (n_rows,)

    fused_add_layernorm_kernel_2048[grid](
        x_ptr=x,
        pos_ptr=pos_embed,
        weight_ptr=weight,
        bias_ptr=bias,
        out_add_ptr=out_add,
        out_norm_ptr=out_norm,
        n_rows=n_rows,
        eps=1e-05,
        BLOCK_SIZE=2048,
    )

    return (out_add, out_norm)


def replacement_func():
    return fused_add_layernorm_2048