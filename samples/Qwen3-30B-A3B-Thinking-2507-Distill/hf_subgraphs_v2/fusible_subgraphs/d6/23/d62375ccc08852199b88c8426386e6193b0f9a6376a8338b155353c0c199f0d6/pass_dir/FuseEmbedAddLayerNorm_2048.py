import torch
from pass_dir.shared_kernels import dispatch_layernorm


def pattern(x, weight, bias):
    """
    Match layer_norm(normalized_shape=(2048,)).
    'x' matches the output of the preceding add operation.
    """
    tmp_14 = torch.nn.functional.layer_norm(x, (2048,), weight, bias, 1e-05)
    return tmp_14


def replacement_args(x, weight, bias):
    return (x, weight, bias, "2048")


def replacement_func():
    return dispatch_layernorm