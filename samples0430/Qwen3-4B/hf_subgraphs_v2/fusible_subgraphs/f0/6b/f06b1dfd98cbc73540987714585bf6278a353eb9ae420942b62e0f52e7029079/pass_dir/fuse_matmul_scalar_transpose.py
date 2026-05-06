import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def triton_compute_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out1_ptr,
    out2_ptr,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= 2:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    in2_row = tl.load(in_2_ptr + pid * N, offsets, mask=mask, other=0.0)
    in1_col = tl.load(in_1_ptr, offsets, mask=mask, other=0.0)
    dot_prod = tl.zeros(tl.float32)
    for i in range(BLOCK_SIZE):
        dot_prod += in2_row[i] * in1_col[i]
    in0 = tl.load(in_0_ptr)
    result = dot_prod * in0
    tl.store(out1_ptr + pid, result)
    tl.store(out2_ptr + pid, result)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    N = 1024
    out1 = torch.empty(2, device=in_0.device, dtype=in_0.dtype)
    out2 = torch.empty(1, 2, device=in_0.device, dtype=in_0.dtype)
    triton_compute_kernel[(2,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out1_ptr=out1,
        out2_ptr=out2,
        N=N,
        BLOCK_SIZE=1024,
    )
    return (out1, out2)

def replacement_func():
    return kernel_wrapper