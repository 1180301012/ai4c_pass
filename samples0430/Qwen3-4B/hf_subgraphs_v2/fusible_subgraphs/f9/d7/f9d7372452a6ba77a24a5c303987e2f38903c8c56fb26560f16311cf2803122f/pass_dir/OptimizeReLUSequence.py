import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2, in3):
    add1 = in3 + in0
    add2 = add1 + in2
    relu_out = torch.nn.functional.relu(add2)
    view_out = in1.view(1, 32, -1)
    permute_out = view_out.permute(0, 2, 1)
    return (relu_out, permute_out)

def replacement_args(in0, in1, in2, in3):
    return (in0, in1, in2, in3)

@triton.jit
def triton_add_relu_kernel(
    in0_ptr,
    in2_ptr,
    in3_ptr,
    out_relu_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (i < n_elements)
    in0 = tl.load(in0_ptr + i, mask=mask)
    in2 = tl.load(in2_ptr + i, mask=mask)
    in3 = tl.load(in3_ptr + i, mask=mask)
    add1 = in3 + in0
    add2 = add1 + in2
    relu_out = tl.where(add2 > 0, add2, 0.0)
    tl.store(out_relu_ptr + i, relu_out, mask=mask)

@torch.fx.wrap
def triton_add_relu(in0, in1, in2, in3):
    N = in0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out_relu = torch.empty_like(in0)
    triton_add_relu_kernel[(num_programs,)](
    in0_ptr=in0,
    in2_ptr=in2,
    in3_ptr=in3,
    out_relu_ptr=out_relu,
    n_elements=N,
    BLOCK_SIZE=BLOCK_SIZE,
)
    view_out = in1.view(1, 32, -1)
    permute_out = view_out.permute(0, 2, 1)
    return (out_relu, permute_out)

def replacement_func():
    return triton_add_relu