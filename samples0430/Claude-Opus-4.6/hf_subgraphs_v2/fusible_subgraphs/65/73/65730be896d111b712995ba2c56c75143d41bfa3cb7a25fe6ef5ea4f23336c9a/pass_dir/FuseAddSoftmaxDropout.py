import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_1 += in_0
    tmp_1 = in_1.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    row_size,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < row_size

    row_start = row_idx * row_size

    # Load inputs
    in_0 = tl.load(in_0_ptr + row_start + col_offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Add and cast to float32
    x = (in_0 + in_1).to(tl.float32)

    # Set masked positions to -inf for correct softmax
    x = tl.where(mask, x, float('-inf'))

    # Softmax in float32
    max_val = tl.max(x, axis=0)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Store (automatically casts to output dtype)
    tl.store(out_ptr + row_start + col_offsets, softmax_out, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    shape = in_0.shape
    row_size = shape[-1]
    n_rows = in_0.numel() // row_size

    out = torch.empty_like(in_0)

    BLOCK_SIZE = triton.next_power_of_2(row_size)

    grid = (n_rows,)
    fused_add_softmax_kernel[grid](
        in_0, in_1, out,
        row_size, n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add_softmax