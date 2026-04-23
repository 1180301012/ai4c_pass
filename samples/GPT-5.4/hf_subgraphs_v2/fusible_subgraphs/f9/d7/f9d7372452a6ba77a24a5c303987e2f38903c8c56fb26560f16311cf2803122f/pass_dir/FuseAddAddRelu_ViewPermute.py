import torch
import triton
import triton.language as tl
from pass_dir.fx_pattern_helper import build_iadd_relu_pattern


pattern = build_iadd_relu_pattern()


def replacement_args(acc, in_2):
    return (acc, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit

def _fused_add_add_relu_kernel(
    acc_ptr,
    x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    acc = tl.load(acc_ptr + offsets, mask=mask)
    b = tl.load(x2_ptr + offsets, mask=mask)

    tmp = (acc + b).to(acc.dtype)
    zero = tl.zeros((BLOCK_SIZE,), dtype=tmp.dtype)
    out = tl.maximum(tmp, zero)

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(acc, in_2):
    out = torch.empty_like(acc)
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_add_add_relu_kernel[grid](
        acc,
        in_2,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_add_add_relu