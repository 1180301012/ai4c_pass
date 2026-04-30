import torch
from pass_dir.bn_inference_only_shared import replacement_func


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4)