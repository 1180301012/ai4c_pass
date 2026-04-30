import torch
import triton
import triton.language as tl


# Pattern matching function - must match model.py exactly (without cleanup statements)
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return (tmp_8,)


def replacement_args(in_0):
    return (in_0,)


# Combine function for associative scan (prefix sum)
@triton.jit
def add_combine(a, b):
    return a + b


@triton.jit
def fused_ne_cumsum_mask_add1_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_cols

    # Load input values (out-of-bounds loaded as 1, treated as padding)
    vals = tl.load(input_ptr + row_idx * n_cols + offsets, mask=valid, other=1)

    # Compute mask as int64: 1 where != 1 (real token), 0 where == 1 (padding)
    ne_mask_int = (vals != 1).to(tl.int64)

    # Inclusive prefix sum using associative scan
    cumsum = tl.associative_scan(ne_mask_int, 0, add_combine)

    # Final result: cumsum * ne_mask_int + 1
    # When val == 1 (padding): 0 * 0 + 1 = 1
    # When val != 1 (real token): cumsum * 1 + 1 = cumsum + 1
    result = cumsum * ne_mask_int + 1

    # Store result (only in-bounds positions)
    tl.store(output_ptr + row_idx * n_cols + offsets, result, mask=valid)


@torch.fx.wrap
def fused_ne_cumsum_mask_add1(in_0):
    n_rows, n_cols = in_0.shape
    output = torch.empty_like(in_0)

    # Choose BLOCK_SIZE: next power of 2 >= n_cols, with minimum of 16
    BLOCK_SIZE = triton.next_power_of_2(max(n_cols, 16))

    grid = (n_rows,)

    fused_ne_cumsum_mask_add1_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_ne_cumsum_mask_add1