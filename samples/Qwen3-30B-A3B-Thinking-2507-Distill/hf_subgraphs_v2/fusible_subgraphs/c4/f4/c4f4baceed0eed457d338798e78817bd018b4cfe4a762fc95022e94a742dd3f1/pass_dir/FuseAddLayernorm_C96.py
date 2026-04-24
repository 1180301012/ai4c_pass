import torch
from pass_dir.shared_layernorm import dispatch_fused_view_add_ln


def pattern(x, y, weight, bias):
    """
    Match: view(1,16384,96) + add + layer_norm(96).
    x = tmp_7 (already window-partitioned, [1,16384,96])
    y = in_2 (addend)
    Returns (added, ln_out) matching the model's (tmp_8, tmp_9).
    """
    tmp_7 = x.view(1, 16384, 96)
    tmp_8 = y + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), weight, bias, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, "C96")


def replacement_func():
    return dispatch_fused_view_add_ln