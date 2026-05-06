import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    bmm = torch.bmm(in_0, in_1)
    tmp1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp2 = torch.nn.functional.dropout(tmp1, p=0.0, training=False)
    bmm1 = torch.bmm(tmp2, in_2)
    tmp4 = bmm1.view(1, bmm1.shape[1], 1, bmm1.shape[2])
    tmp5 = tmp4.transpose(1, 2)
    tmp6 = tmp5.reshape(1, 1, bmm1.shape[1] * bmm1.shape[2])
    return (tmp6,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    bmm1_ptr, out_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offset = row * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        if offset + i < n_elements:
            tl.store(out_ptr + offset + i, tl.load(bmm1_ptr + offset + i))

@torch.fx.wrap
def kernel_wrapper(bmm1):
    N = bmm1.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty(N, device=bmm1.device, dtype=bmm1.dtype)
    optimized_kernel[(num_programs,)](\
        bmm1_ptr=bmm1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return kernel_wrapper