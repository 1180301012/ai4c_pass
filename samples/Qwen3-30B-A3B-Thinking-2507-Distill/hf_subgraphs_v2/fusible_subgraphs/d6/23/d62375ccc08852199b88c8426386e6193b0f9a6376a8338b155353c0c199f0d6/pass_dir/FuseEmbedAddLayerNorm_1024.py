import torch
from pass_dir.shared_kernels import dispatch_layernorm


def pattern(x, weight, bias):
    """
    Match layer_norm(normalized_shape=(1024,)).
    'x' matches the output of the preceding add operation.
    """
    tmp_14 = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return tmp_14


def replacement_args(x, weight, bias):
    return (x, weight, bias, "1024")


def replacement_func():
    return dispatch_layernorm