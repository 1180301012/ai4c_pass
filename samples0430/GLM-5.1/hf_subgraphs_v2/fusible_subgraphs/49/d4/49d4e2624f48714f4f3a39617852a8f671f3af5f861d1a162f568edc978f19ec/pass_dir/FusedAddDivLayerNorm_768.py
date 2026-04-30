import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_div_layernorm_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    offsets = row_start + col_offsets

    # Load inputs - use eviction policies to optimize cache usage
    # in2 and in3 are consumed early, evict from cache first
    in2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
    in3 = tl.load(in3_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

    # Fused: add + divide by 2
    x = (in2 + in3) / 2.0

    # Compute mean
    x_mean = tl.sum(x, axis=0) / n_cols

    # Compute centered values - reuse for variance and normalization
    x_centered = tl.where(mask, x - x_mean, 0.0)

    # Compute variance
    x_var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Load weight and bias - keep in cache since used for normalization
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)

    # Normalize using precomputed centered values
    rstd = 1.0 / tl.sqrt(x_var + eps)
    out = x_centered * rstd * weight + bias

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_div_layernorm(bias, weight, in2, in3):
    n_rows = in2.shape[0]
    n_cols = in2.shape[1]
    eps = 1e-12

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    out = torch.empty_like(in2)

    grid = (n_rows,)

    fused_add_div_layernorm_kernel[grid](
        in2_ptr=in2,
        in3_ptr=in3,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add_div_layernorm