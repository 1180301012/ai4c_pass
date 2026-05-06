import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def optimized_kernel(in_3_ptr, in_1_ptr, in_0_ptr, in_2_ptr, out_ptr, N: tl.int32, C: tl.int32, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, BLOCK_SIZE)
    for j_index in j:
        in_3_val = tl.load(in_3_ptr + i * C + j_index)
        weight_row = tl.load(in_1_ptr + j_index * C)
        dot = tl.dot(in_3_val, weight_row)
        bias = tl.load(in_0_ptr + j_index)
        in_2_val = tl.load(in_2_ptr + i * C + j_index)
        total = in_2_val + dot + bias
        total = tl.where(total > 0, total, 0.0)
        tl.store(out_ptr + i * C + j_index, total)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    N = in_3.shape[0]
    C = in_3.shape[1]
    out = torch.empty((N, C), dtype=in_3.dtype, device=in_3.device)
    BLOCK_SIZE = 128
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[grid](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        N=N,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return kernel_wrapper