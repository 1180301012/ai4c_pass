import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2):
    matmul = torch.matmul(in2, in1)
    tmp1 = matmul * in0
    tmp2 = tmp1.T
    return (tmp1, tmp2)

def replacement_args(in0, in1, in2):
    return (in0, in1, in2)

@triton.jit
def optimized_kernel(
    in2_ptr,
    in1_ptr,
    in0,
    out1_ptr,
    out2_ptr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= num_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 512
    in2_row = tl.load(in2_ptr + i*512 + offsets, mask=mask, other=0.0)
    in1_row = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    sum_val = 0.0
    for j in range(BLOCK_SIZE):
        sum_val += in2_row[j] * in1_row[j]
    result = sum_val * in0

    tl.store(out1_ptr + i, result)
    tl.store(out2_ptr + i, result)

@torch.fx.wrap
def kernel_wrapper(in0, in1, in2):
    num_rows = in2.shape[0]
    out1 = torch.empty((num_rows, 1), device=in2.device, dtype=in2.dtype)
    out2 = torch.empty((1, num_rows), device=in2.device, dtype=in2.dtype)
    BLOCK_SIZE = 128
    grid = (num_rows, )
    optimized_kernel[grid](
        in2_ptr=in2,
        in1_ptr=in1,
        in0=in0,
        out1_ptr=out1,
        out2_ptr=out2,
        num_rows=num_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out1, out2

def replacement_func():
    return kernel_wrapper