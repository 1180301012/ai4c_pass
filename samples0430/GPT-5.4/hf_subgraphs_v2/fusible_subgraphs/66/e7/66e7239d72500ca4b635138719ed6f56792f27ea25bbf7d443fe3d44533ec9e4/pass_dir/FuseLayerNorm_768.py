from pass_dir.shared_fused_ops import shared_runtime_dispatch, shared_replacement_func
import torch


def pattern(in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (None, in_1, in_2, in_3, "layer_norm")


def replacement_impl(in_1, in_2, in_3):
    return shared_runtime_dispatch(None, in_1, in_2, in_3, "layer_norm")


def replacement_func():
    return shared_replacement_func()