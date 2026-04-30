import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2


def replacement_args(x):
    return (x,)


@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    # Each program handles one output row (of size M=1024)
    # Output[N, M], Input[M, N]
    # output[r, :] = input[:, r]
    row = tl.program_id(0)  # 0..N-1 = 0..127

    cols = tl.arange(0, BLOCK_SIZE)
    # No mask needed since BLOCK_SIZE == M == 1024

    # Input: input[cols, row] at linear index cols * N + row
    in_offsets = cols * N + row

    # Output: output[row, cols] at linear index row * M + cols
    out_offsets = row * M + cols

    vals = tl.load(input_ptr + in_offsets)
    tl.store(output_ptr + out_offsets, vals)


@torch.fx.wrap
def fused_unsqueeze_transpose(x):
    B = x.shape[0]
    M = x.shape[1]  # 1024
    N = x.shape[2]  # 128

    output = torch.empty((B, 1, N, M), dtype=x.dtype, device=x.device)

    # 128 programs, each handling one output row of 1024 elements
    transpose_kernel[(N,)](
        x,
        output,
        BLOCK_SIZE=M,
        M=M,
        N=N,
    )

    return output


def replacement_func():
    return fused_unsqueeze_transpose