import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hardtanh = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return hardtanh * conv2d
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr
):
    pass

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch = in_2.shape[0]
    out = torch.empty_like(in_2)
    grid = (batch,)
    optimized_kernel[grid](
        a_ptr=in_2,
        b_ptr=in_1,
        c_ptr=in_0,
        d_ptr=in_3,
        out_ptr=out,
        N=in_2.shape[1],
        M=in_2.shape[2],
        K=in_2.shape[3]
    )
    return out
def replacement_func():
    return kernel_wrapper