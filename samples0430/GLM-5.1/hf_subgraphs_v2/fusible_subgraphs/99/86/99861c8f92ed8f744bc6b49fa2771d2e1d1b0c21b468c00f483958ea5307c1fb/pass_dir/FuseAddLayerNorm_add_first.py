import torch
import triton
import triton.language as tl


@triton.jit
def triton_ln_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_val = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(tl.where(mask, input_val, 0.0), axis=0) / n_cols
    diff = input_val - mean
    var = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    out = weight * normalized + bias_val

    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


@torch.fx.wrap
def triton_layer_norm(input, normalized_shape, weight, bias, eps):
    n_rows = input.numel() // input.shape[-1]
    n_cols = input.shape[-1]

    out = torch.empty_like(input)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    triton_ln_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        n_cols=n_cols, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def pattern(input, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


def replacement_args(input, normalized_shape, weight, bias, eps):
    return (input, normalized_shape, weight, bias, eps)


def replacement_func():
    return triton_layer_norm