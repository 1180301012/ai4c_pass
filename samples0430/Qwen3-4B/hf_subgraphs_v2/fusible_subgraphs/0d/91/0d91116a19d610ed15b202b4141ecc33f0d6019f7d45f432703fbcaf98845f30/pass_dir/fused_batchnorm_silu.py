import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6
def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    return (in_5, in_4, in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    x_ptr: tl.ptr[tl.float32],
    y_ptr: tl.ptr[tl.float32],
    mean_ptr: tl.ptr[tl.float32],
    var_ptr: tl.ptr[tl.float32],
    bias_ptr: tl.ptr[tl.float32],
    weight_ptr: tl.ptr[tl.float32],
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pass

@torch.fx.wrap
def fused_op(x, y, mean, var, bias, weight):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    fused_kernel[(num_programs,)](\n        x_ptr=x,
        y_ptr=y,
        mean_ptr=mean,
        var_ptr=var,
        bias_ptr=bias,
        weight_ptr=weight,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return fused_op