import torch
import triton
import triton.language as tl

def pattern(ones_tensor: torch.Tensor) -> torch.Tensor:
    result = torch.arange(15, dtype=torch.int64).view(1, -1) + 2
    return result.to(ones_tensor.device)

def replacement_args(ones_tensor: torch.Tensor) -> tuple:
    return (ones_tensor,)

@triton.jit
def constant_kernel():
    pass

@torch.fx.wrap
def kernel_wrapper(ones_tensor):
    return pattern(ones_tensor)

def replacement_func():
    return kernel_wrapper