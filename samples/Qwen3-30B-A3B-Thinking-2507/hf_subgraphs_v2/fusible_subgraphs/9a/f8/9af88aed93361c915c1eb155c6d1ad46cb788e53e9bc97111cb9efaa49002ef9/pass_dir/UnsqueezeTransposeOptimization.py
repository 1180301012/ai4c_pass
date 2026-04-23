import torch
import triton
import triton.language as tl


def pattern(x):
    tmp1 = x.unsqueeze(1)
    tmp2 = tmp1.transpose(2, 3)
    return tmp2

def replacement_args(x):
    return (x,)

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_block_rows = N // BLOCK_SIZE
    num_block_cols = M // BLOCK_SIZE
    block_row = pid // num_block_cols
    block_col = pid % num_block_cols
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    shared_input = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float16)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            input_idx = (col_start + j) * N + (row_start + i)
            shared_input[i, j] = tl.load(input_ptr + input_idx)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            output_idx = (row_start + i) * M + (col_start + j)
            tl.store(output_ptr + output_idx, shared_input[i, j])

@torch.fx.wrap
def transpose_wrapper(x):
    M = x.shape[2]
    N = x.shape[3]
    out = torch.empty(x.shape[0], x.shape[1], N, M, dtype=x.dtype, device=x.device)
    block_size = 32
    num_blocks = (N + block_size - 1) // block_size * (M + block_size - 1) // block_size
    transpose_kernel[(num_blocks,)](
        x.data_ptr(),
        out.data_ptr(),
        M,
        N,
        block_size
    )
    return out

def replacement_func():
    return transpose_wrapper