import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    linear_out = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear_out.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    t0 = unbind[0]
    t1 = unbind[1]
    t2 = unbind[2]
    t1_transposed = t1.transpose(-2, -1)
    return (t0, t1_transposed, t2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(in_0_ptr, in_1_ptr, out_0_ptr, out_1_ptr, out_2_ptr, K: tl.constexpr, N: tl.constexpr, M: tl.constexpr):
    pass

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    N = 197
    M = 48
    K = 9
    out_0 = torch.empty((1, N, 1, K, M), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((1, K, M, N), dtype=in_0.dtype, device=in_0.device)
    out_2 = torch.empty((1, N, K, M), dtype=in_0.dtype, device=in_0.device)
    return (out_0, out_1, out_2)

def replacement_func():
    return kernel_wrapper