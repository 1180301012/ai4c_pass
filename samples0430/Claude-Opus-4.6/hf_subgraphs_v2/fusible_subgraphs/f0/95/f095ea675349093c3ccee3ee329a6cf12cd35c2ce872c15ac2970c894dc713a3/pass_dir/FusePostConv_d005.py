import torch
import triton
import triton.language as tl


def pattern(conv_result, residual):
    sliced = conv_result[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    gelu_out = torch.nn.functional.gelu(sliced)
    transposed = gelu_out.transpose(1, 2)
    added = residual + transposed
    return added


def replacement_args(conv_result, residual):
    return (conv_result, residual)


@triton.jit
def fused_gelu_transpose_add_kernel(
    conv_ptr, residual_ptr, out_ptr,
    T, C: tl.constexpr, T_conv,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    t = row_idx % T
    b = row_idx // T

    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load conv_result[b, c, t] (strided access: stride = T_conv between channels)
    conv_base = b * C * T_conv + t
    conv_vals = tl.load(conv_ptr + conv_base + col_offsets * T_conv).to(tl.float32)

    # GELU
    gelu_vals = conv_vals * 0.5 * (1.0 + tl.math.erf(conv_vals * 0.7071067811865476))

    # Load residual[b, t, c] (contiguous access)
    res_base = b * T * C + t * C
    res_vals = tl.load(residual_ptr + res_base + col_offsets).to(tl.float32)

    # Add and store
    tl.store(out_ptr + res_base + col_offsets, res_vals + gelu_vals)


@torch.fx.wrap
def fused_gelu_transpose_add(conv_result, residual):
    B = conv_result.shape[0]
    C = conv_result.shape[1]
    T_conv = conv_result.shape[2]
    T = T_conv - 1

    out = torch.empty((B, T, C), dtype=residual.dtype, device=residual.device)

    grid = (B * T,)
    fused_gelu_transpose_add_kernel[grid](
        conv_result, residual, out,
        T, C, T_conv,
        BLOCK_SIZE=C,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_gelu_transpose_add