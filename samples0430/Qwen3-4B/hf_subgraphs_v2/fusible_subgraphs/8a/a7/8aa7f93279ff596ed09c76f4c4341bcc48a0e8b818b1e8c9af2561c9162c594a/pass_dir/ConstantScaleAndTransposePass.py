import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    return (in_1 * 0.1767766952966369, in_0.transpose(-2, -1))

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    in0_shape,
    in1_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pass

@torch.fx.wrap
def kernel_wrapper(in0, in1):
    B, C, H, W = in0.shape[0], in0.shape[1], in0.shape[2], in0.shape[3]
    out0 = torch.empty_like(in0)
    out1 = torch.empty_like(in0)
    optimized_kernel[(1,)].__call__(
        in0_ptr=in0,
        in1_ptr=in1,
        out0_ptr=out0,
        out1_ptr=out1,
        in0_shape=(B, C, H, W),
        in1_shape=(B, C, H, W),
        BLOCK_SIZE=128,
    )
    return (out0, out1)

def replacement_func():
    return kernel_wrapper