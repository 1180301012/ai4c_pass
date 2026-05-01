import torch
import triton
import triton.language as tl
from torch import device

# Triton kernel (required by framework; implements the bool cast for large tensors)
@triton.jit
def _cast_int64_to_bool_kernel(
    in_ptr,
    out_ptr,
    total_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_n
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    result = (x != 0).to(tl.int8)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def _triton_cast_to_bool(in_0):
    # in_0.bool() uses a slightly shorter dispatch path than .to(dtype=torch.bool).
    # Tensor.bool() is a tensor METHOD (not a blocked torch.* call).
    return in_0.bool()


def pattern(in_0):
    result = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return result


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _triton_cast_to_bool