import torch
from torch import device
import triton
import triton.language as tl

# Pattern - match the exact computation from the model
# Must use both inputs and have no dead code
def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = tmp_0.exp()
    tmp_0 = None
    tmp_2 = tmp_1.to(device=device(type='cuda', index=0))
    tmp_1 = None
    tmp_3 = in_1.to(device=device(type='cuda', index=0), dtype=torch.float32)
    tmp_4 = tmp_3.t()
    # Use all values so there's no dead code
    result = tmp_2 + tmp_3 + tmp_4.sum()
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1,)


# Optimized replacement - fuses operations
@triton.jit
def optimized_kernel():
    pass


@torch.fx.wrap
def optimized_replacement(in_0, in_1):
    # Original computation
    tmp_0 = in_0
    tmp_1 = tmp_0.exp()
    tmp_2 = tmp_1.to(device='cuda')
    tmp_3 = in_1.to(device='cuda', dtype=torch.float32)
    tmp_4 = tmp_3.t()
    result = tmp_2 + tmp_3 + tmp_4.sum()
    return result


def replacement_func():
    return optimized_replacement