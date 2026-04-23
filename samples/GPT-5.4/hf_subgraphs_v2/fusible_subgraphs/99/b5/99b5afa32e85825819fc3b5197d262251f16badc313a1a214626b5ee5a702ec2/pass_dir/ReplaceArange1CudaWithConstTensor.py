import torch
import triton
import triton.language as tl


def pattern():
    tmp_0 = torch.arange(1, device=torch.device(type='cuda', index=0))
    tmp_1 = torch._functorch.vmap.lazy_load_decompositions()
    return (tmp_0,)


def replacement_args():
    return tuple()


@triton.jit
def _fill_zero_kernel(out_ptr):
    tl.store(out_ptr, 0)


@torch.fx.wrap
def _replace_arange1_cuda_with_const_tensor():
    # Use only allowed tensor allocation APIs in wrapper.
    # arange(1) always returns a length-1 int64 tensor containing 0.
    out = torch.zeros((1,), device='cuda', dtype=torch.int64)
    return (out,)



def replacement_func():
    return _replace_arange1_cuda_with_const_tensor