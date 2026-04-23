import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch_ops import dispatch_shared


def pattern(tmp_10):
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=torch.device(type='cuda'))
    return tmp_11


def replacement_args(tmp_10):
    return (tmp_10, 'ones_f32')


def replacement_func():
    return dispatch_shared