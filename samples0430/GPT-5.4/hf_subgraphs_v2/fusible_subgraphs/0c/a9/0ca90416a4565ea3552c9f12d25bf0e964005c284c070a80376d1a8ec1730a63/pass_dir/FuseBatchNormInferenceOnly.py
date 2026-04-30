import torch
from pass_dir.bn_inference_only_shared import replacement_func


def pattern(running_mean, running_var, bias, weight, x):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)


def replacement_args(running_mean, running_var, bias, weight, x):
    return (running_mean, running_var, bias, weight, x)