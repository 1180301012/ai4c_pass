"""
Pass for RECT_L bfloat16 graph: arange(0, 128) -> view(1, -1) -> repeat(2, 1)
Fuses the three operations into a single Triton kernel.
"""
import torch
from torch import device as torch_device
from pass_dir.shared_kernels import fused_arange_view_repeat_dispatcher


def pattern():
    """Pattern that matches arange -> view -> repeat chain for RECT_L bfloat16 (128 elements)"""
    tmp_0 = torch.arange(0, 128, device=torch_device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    """Extract arguments for RECT_L bfloat16: n_elements=128, repeat_dim0=2"""
    return (128, 2, 'bfloat16', "cuda", "route_rect_bf16")


def replacement_func():
    """Returns the shared dispatcher function"""
    return fused_arange_view_repeat_dispatcher