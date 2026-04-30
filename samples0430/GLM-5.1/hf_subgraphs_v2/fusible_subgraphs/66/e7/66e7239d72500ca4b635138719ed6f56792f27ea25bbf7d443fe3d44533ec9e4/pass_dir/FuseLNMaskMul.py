import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


@triton.jit
def layer_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAST_DTYPE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # Load input row and cast to float32
    input_row = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(input_row, axis=0) / n_cols

    # Compute variance
    diff = input_row - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Load weight and bias (cast to float32)
    weight_row = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    bias_row = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Apply weight and bias
    output = normalized * weight_row + bias_row

    # Store output (cast back to input dtype)
    if CAST_DTYPE == 0:
        tl.store(output_ptr + row_start + col_offsets, output.to(tl.float16), mask=col_mask)
    else:
        tl.store(output_ptr + row_start + col_offsets, output.to(tl.bfloat16), mask=col_mask)


@torch.fx.wrap
def triton_layer_norm(in_1, in_2, in_3):
    input_dtype = in_3.dtype
    n_rows = in_3.shape[0] * in_3.shape[1]
    n_cols = in_3.shape[2]

    output = torch.empty_like(in_3)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    eps = 1e-12

    if input_dtype == torch.float16:
        CAST_DTYPE = 0
    elif input_dtype == torch.bfloat16:
        CAST_DTYPE = 1
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")

    grid = (n_rows,)

    layer_norm_kernel[grid](
        in_3, in_2, in_1, output,
        n_rows, n_cols,
        eps=eps, BLOCK_SIZE=BLOCK_SIZE, CAST_DTYPE=CAST_DTYPE,
    )

    return output


def replacement_func():
    return triton_layer_norm