import torch
import triton
import triton.language as tl

# Pattern: first op is aten.add_.Tensor (in_3 += in_0 on a placeholder → ATen op),
#          second op is iadd (in_4 += in_2 on a computed result → operator.iadd).
# The += on a *result* proxy traces as iadd, not add_.Tensor.
def pattern(in_0, in_2, in_3):
    in_4 = torch.ops.aten.add_.Tensor(in_3, in_0)
    in_4 += in_2          # result proxy += → should trace as iadd
    tmp_2 = torch.nn.functional.relu(in_4, inplace=True)
    return tmp_2


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_add_relu_kernel(
    in0_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x0 = tl.load(in0_ptr + offsets, mask=mask)
    x2 = tl.load(in2_ptr + offsets, mask=mask)
    x3 = tl.load(in3_ptr + offsets, mask=mask)
    res = x3 + x0 + x2
    zeros = tl.zeros_like(res)
    res = tl.where(res > zeros, res, zeros)
    tl.store(out_ptr + offsets, res, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_add_add_relu_kernel[grid](in_0, in_2, in_3, out, n_elements=N)
    return out


def replacement_func():
    return fused_add_add_relu