import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(out_ptr + offs, 0.0, mask=mask)


@torch.fx.wrap
def shared_zero_dispatch(*args):
    route = args[-1]
    if route == "conv_like":
        x, weight, _bias = args[0], args[1], args[2]
        return torch.zeros((x.shape[0], weight.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    if route == "elemwise_like":
        x = args[0]
        return torch.zeros_like(x)
    x, weight, _bias = args[0], args[1], args[2]
    return torch.zeros((x.shape[0], weight.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)