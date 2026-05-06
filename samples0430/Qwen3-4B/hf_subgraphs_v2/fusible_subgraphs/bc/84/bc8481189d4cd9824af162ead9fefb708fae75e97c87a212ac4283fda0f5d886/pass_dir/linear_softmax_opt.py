import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim = 1)
    return tmp_4
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pass

def optimized_kernel_wrapper(in_0, in_1, in_2):
    n_elements = in_2.numel()
    out = torch.empty(n_elements, device=in_2.device, dtype=in_2.dtype)
    BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel[(num_programs,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return optimized_kernel_wrapper