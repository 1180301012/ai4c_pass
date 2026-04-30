import torch
import triton
import triton.language as tl

from pass_dir.shared_simple_routes import shared_simple_route


def pattern(x):
    y = torch.nn.functional.dropout(x, 0.0, False, False)
    return (y,)


def replacement_args(x):
    return (x, 'dropout_identity')


def replacement_func():
    return shared_simple_route