import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


@triton.jit
def triton_layernorm_kernel(
    in_3_ptr, in_2_ptr, in_1_ptr,
    out_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Load input row
    row_start = row_idx * stride_row
    x = tl.load(in_3_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Load weight and bias
    weight = tl.load(in_2_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(in_1_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute layer norm
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + 1e-12)
    x_norm = x_centered * inv_std
    out = x_norm * weight + bias

    # Store output
    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


@torch.fx.wrap
def triton_layernorm(in_1, in_2, in_3):
    B = in_3.shape[0]
    S = in_3.shape[1]
    D = in_3.shape[2]
    num_rows = B * S

    out = torch.empty_like(in_3)

    triton_layernorm_kernel[(num_rows,)](
        in_3, in_2, in_1,
        out,
        D,
        D=D,
        BLOCK_SIZE=1024,
        num_warps=4,
    )

    return out


def replacement_func():
    return triton_layernorm