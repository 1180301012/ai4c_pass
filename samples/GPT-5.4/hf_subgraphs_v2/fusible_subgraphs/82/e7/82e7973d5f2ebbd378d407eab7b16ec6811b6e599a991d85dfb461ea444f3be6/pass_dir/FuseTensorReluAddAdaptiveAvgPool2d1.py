import torch

from pass_dir.shared_fused_relu_add_gap import replacement_func


def pattern(in_0, in_1):
    tmp_0 = in_1.relu()
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "relu_add_gap_tensor_relu")