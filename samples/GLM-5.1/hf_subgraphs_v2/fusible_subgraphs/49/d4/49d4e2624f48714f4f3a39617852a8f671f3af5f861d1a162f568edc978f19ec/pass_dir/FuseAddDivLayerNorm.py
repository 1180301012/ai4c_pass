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
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs
    in_2 = tl.load(in_2_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)

    # Fuse: add + divide by 2
    x = (in_2 + in_3) / 2.0

    # Layer norm: compute mean
    x_mean = tl.sum(x, axis=0) / n_cols
    # Compute variance
    x_centered = x - x_mean
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    # Normalize
    x_normed = x_centered / tl.sqrt(variance + eps)

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    # Apply affine transform
    out = x_normed * weight + bias

    # Store
    tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 768}, num_warps=4),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_add_div_layernorm_kernel_autotuned(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs
    in_2 = tl.load(in_2_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)

    # Fuse: add + divide by 2
    x = (in_2 + in_3) / 2.0

    # Layer norm: compute mean
    x_mean = tl.sum(x, axis=0) / n_cols
    # Compute variance
    x_centered = x - x_mean
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    # Normalize
    x_normed = x_centered / tl.sqrt(variance + eps)

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    # Apply affine transform
    out = x_normed * weight + bias

    # Store
    tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_div_layernorm(in_0, in_1, in_2, in_3):
    assert in_2.shape == in_3.shape
    n_rows = in_2.shape[0]
    n_cols = in_2.shape[1]
    eps = 1e-12

    out = torch.empty_like(in_2)

    # Use the non-autotuned version for small sizes, autotuned for larger
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    fused_add_div_layernorm_kernel[grid](
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


def replacement_func():
    return fused_add_div_layernorm