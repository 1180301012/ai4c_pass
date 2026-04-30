import torch
from pass_dir.shared_fused_pool_bn_silu import shared_replacement_func


def pattern(running_mean, running_var, bias, weight, x):
    tmp_4 = torch.ops.aten.reshape.default(x, [1, 512, 16, 16])
    tmp_5 = torch.ops.aten.avg_pool2d.default(tmp_4, [2, 2], [2, 2], [0, 0], False, True, None)
    tmp_6 = torch.ops.aten._native_batch_norm_legit_no_training.default(tmp_5, weight, bias, running_mean, running_var, 0.1, 1e-05)
    tmp_7 = torch.ops.aten.silu.default(tmp_6[0])
    return tmp_7


def replacement_args(running_mean, running_var, bias, weight, x):
    return (running_mean, running_var, bias, weight, x, "full_static")


def replacement_func():
    return shared_replacement_func()