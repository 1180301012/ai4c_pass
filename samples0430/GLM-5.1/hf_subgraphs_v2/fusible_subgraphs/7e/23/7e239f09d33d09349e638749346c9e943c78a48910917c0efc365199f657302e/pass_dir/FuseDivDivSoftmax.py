import torch
import triton
import triton.language as tl


def pattern(x):
    result = x.softmax(dim=-1)
    return (result,)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['row_len'],
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    num_rows, row_len,
    stride_row_in, stride_col_in,
    stride_row_out, stride_col_out,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_in = input_ptr + row_idx * stride_row_in
    row_start_out = output_ptr + row_idx * stride_row_out

    # Online softmax: 2-pass algorithm
    # Pass 1: compute row_max and row_sum using online algorithm
    row_max = -float('inf')
    row_sum = 0.0
    
    for start in range(0, row_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        x = tl.load(row_start_in + offsets * stride_col_in, mask=mask, other=-float('inf')).to(tl.float32)
        
        # Online max update
        new_max = tl.maximum(row_max, tl.max(x, axis=0))
        
        # Correction factor for sum: when max increases, previous exp values need rescaling
        correction = tl.exp(row_max - new_max)
        row_sum = row_sum * correction + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # Pass 2: normalize and store
    for start in range(0, row_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        x = tl.load(row_start_in + offsets * stride_col_in, mask=mask, other=-float('inf')).to(tl.float32)
        exp_val = tl.exp(x - row_max)
        out = exp_val / row_sum
        tl.store(row_start_out + offsets * stride_col_out, out, mask=mask)


@torch.fx.wrap
def triton_softmax(input_tensor):
    N = input_tensor.shape[-1]
    total_rows = input_tensor.numel() // N

    output = torch.empty_like(input_tensor)

    stride_row_in = N
    stride_col_in = 1
    stride_row_out = N
    stride_col_out = 1

    grid = (total_rows,)

    softmax_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        num_rows=total_rows,
        row_len=N,
        stride_row_in=stride_row_in,
        stride_col_in=stride_col_in,
        stride_row_out=stride_row_out,
        stride_col_out=stride_col_out,
    )

    return output


def replacement_func():
    return triton_softmax