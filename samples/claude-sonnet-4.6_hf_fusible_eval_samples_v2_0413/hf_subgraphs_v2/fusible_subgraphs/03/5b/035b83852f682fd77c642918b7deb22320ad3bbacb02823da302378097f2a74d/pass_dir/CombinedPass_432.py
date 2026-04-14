import torch
from pass_dir.layer_norm_triton_kernel import triton_layer_norm_dispatch


def pattern(x, weight, bias):
    """
    Matches the full forward computation for D=432 models:
    layer_norm (D=432) + position-tensor construction → returns (pos, ln).
    Having x/weight/bias as anchors lets the FX matcher find the subgraph.
    """
    tmp_2 = torch.nn.functional.layer_norm(x, (432,), weight, bias, 1e-06)
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    return (tmp_3, tmp_2)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "combined_432")


def replacement_func():
    return triton_layer_norm_dispatch