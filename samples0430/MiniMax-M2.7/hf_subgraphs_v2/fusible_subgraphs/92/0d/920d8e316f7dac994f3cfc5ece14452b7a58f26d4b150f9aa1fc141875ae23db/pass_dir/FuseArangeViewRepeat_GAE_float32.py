"""
Pass for GAE float32 graph: arange(0, 1000) -> view(1, -1) -> repeat(2, 1)
Fuses the three operations into a single Triton kernel.
"""
import torch
from torch import device as torch_device
from pass_dir.shared_kernels import fused_arange_view_repeat_dispatcher


def pattern():
    """Pattern that matches arange -> view -> repeat chain for GAE float32 (1000 elements)"""
    tmp_0 = torch.arange(0, 1000, device=torch_device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    """Extract arguments for GAE float32: n_elements=1000, repeat_dim0=2"""
    return (1000, 2, 'float32', "cuda", "route_gae_float32")


def replacement_func():
    """Returns the shared dispatcher function"""
    return fused_arange_view_repeat_dispatcher