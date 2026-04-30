import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['D'],
)
@triton.jit
def fused_add_layernorm_kernel_v2(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    sum_out_ptr, norm_out_ptr,
    N, D: tl.constexpr, eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, D)

    base_offset = row_idx * D + col_offsets

    # Load inputs
    in_2 = tl.load(in_2_ptr + base_offset)
    in_3 = tl.load(in_3_ptr + base_offset)

    # Compute sum
    x = in_2 + in_3

    # Store sum output
    tl.store(sum_out_ptr + base_offset, x)

    # Compute layer norm in float32 for accuracy
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / D
    centered = x_f32 - mean
    var = tl.sum(centered * centered, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets)
    bias = tl.load(bias_ptr + col_offsets)

    # Apply normalization
    normed = centered * inv_std
    result = normed * weight.to(tl.float32) + bias.to(tl.float32)

    # Store normalized output (cast back to input dtype)
    tl.store(norm_out_ptr + base_offset, result.to(x.dtype))


@torch.fx.wrap
def fused_add_layernorm_norm_sum(in_0, in_1, in_2, in_3):
    shape = in_2.shape
    D = shape[-1]
    N = in_2.numel() // D

    sum_out = torch.empty_like(in_2)
    norm_out = torch.empty_like(in_2)

    grid = (N,)
    fused_add_layernorm_kernel_v2[grid](
        in_2, in_3, in_1, in_0,
        sum_out, norm_out,
        N, D, 1e-05,
    )

    return (norm_out, sum_out)


def replacement_func():
    return fused_add_layernorm_norm_sum