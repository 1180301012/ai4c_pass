import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_cumsum_mul_sub_long_slice_add_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len,
    MAX_SEQ_LEN: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    row_offset = row_idx * seq_len
    cumsum = 0
    cumsum = cumsum.to(tl.int64)

    for i in range(MAX_SEQ_LEN):
        val = tl.load(input_ptr + row_offset + i)
        cumsum = cumsum + val
        result = cumsum * val + 1
        tl.store(output_ptr + row_offset + i, result)

@torch.fx.wrap
def kernel_wrapper(in_0):
    batch_size, seq_len = in_0.shape

    out = torch.empty_like(in_0)

    if seq_len == 0:
        return out

    # Use exact seq_len to minimize loop iterations and avoid conditional checks
    MAX_SEQ_LEN = seq_len

    grid = (batch_size,)

    fused_cumsum_mul_sub_long_slice_add_kernel[grid](
        input_ptr=in_0,
        output_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
    )

    return out

def replacement_func():
    return kernel_wrapper