import torch
from pass_dir.shared_kernels import bn_wrapped


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match inference-mode batch_norm.
    in_0 = running_mean, in_1 = running_var,
    in_2 = bias (beta),  in_3 = weight (gamma),
    in_4 = input tensor
    """
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # bn_wrapped signature: (input, running_mean, running_var, bias, weight)
    return (in_4, in_0, in_1, in_2, in_3)


def replacement_func():
    return bn_wrapped