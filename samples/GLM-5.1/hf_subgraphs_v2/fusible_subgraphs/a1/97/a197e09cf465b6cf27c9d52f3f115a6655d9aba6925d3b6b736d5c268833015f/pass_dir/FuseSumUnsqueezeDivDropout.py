import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def row_normalize_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load entire row
    row = tl.load(input_ptr + row_idx * input_row_stride + offsets, mask=mask, other=0.0)

    # Compute row sum (accumulate in input dtype to match PyTorch behavior)
    row_sum = tl.sum(row, axis=0)

    # Normalize: divide each element by row sum
    normalized = row / row_sum

    # Store result
    tl.store(output_ptr + row_idx * output_row_stride + offsets, normalized, mask=mask)


@torch.fx.wrap
def row_normalize(input_tensor):
    # Number of rows to normalize (all dims except last)
    n_rows = input_tensor.shape[:-1].numel()
    # Length of each row (last dimension)
    n_cols = input_tensor.shape[-1]

    # Stride between consecutive rows
    input_row_stride = input_tensor.stride(-2)

    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    output_row_stride = output.stride(-2)

    # BLOCK_SIZE must be power of 2 and >= n_cols
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    if BLOCK_SIZE < n_cols:
        BLOCK_SIZE <<= 1

    grid = (n_rows,)
    row_normalize_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        input_row_stride=input_row_stride,
        output_row_stride=output_row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return row_normalize