import torch
import sys
import os

# Add pass_dir to sys.path for shared imports
pass_dir_path = os.path.dirname(os.path.abspath(__file__))
if pass_dir_path not in sys.path:
    sys.path.insert(0, pass_dir_path)
from shared_dispatch import dispatch_wrapper


def pattern(input, weight, bias):
    return torch.nn.functional.layer_norm(input, (16,), weight, bias, 1e-05)


def replacement_args(input, weight, bias):
    return (input, weight, bias, "layernorm_16")


def replacement_func():
    return dispatch_wrapper