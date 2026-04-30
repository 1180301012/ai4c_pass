import torch
import triton
import triton.language as tl
from torch import device


# Match the observable computation only. Do not include dead cleanup statements.
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return (tmp_0,)


# No dynamic arguments are needed for this zero-input graph.
def replacement_args():
    return ()


@triton.jit
def _store_zero_i64_kernel(out_ptr):
    tl.store(out_ptr, tl.full((), 0, tl.int64))


@torch.fx.wrap
def _const_arange1_cuda():
    out = torch.empty((1,), device='cuda', dtype=torch.int64)
    _store_zero_i64_kernel[(1,)](out)
    return (out,)


# Return the function object, not a call.
def replacement_func():
    return _const_arange1_cuda