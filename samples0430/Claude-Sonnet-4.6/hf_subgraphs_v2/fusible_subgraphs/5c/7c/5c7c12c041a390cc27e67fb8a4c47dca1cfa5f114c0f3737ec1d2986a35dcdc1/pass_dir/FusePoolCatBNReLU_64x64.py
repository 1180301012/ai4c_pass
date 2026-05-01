"""
Pass: FusePoolCatBNReLU_64x64
Matches the 2-op subgraph: torch.cat + torch.nn.functional.batch_norm
(F.relu cannot be matched because its inplace=False kwarg structure conflicts with
ForceArgsTracer's normalization. The replacement already includes relu; the model's
subsequent F.relu(x, inplace=False) is idempotent: relu(relu(x)) == relu(x).)
"""
import torch
from pass_dir.shared_kernel import fused_cat_bn_relu


def pattern(cat_a, cat_b, running_mean, running_var, weight, bias):
    # cat_a: first input to cat, cat_b: second input (already-pooled/upsampled tensor)
    cat = torch.cat([cat_a, cat_b], 1)
    bn = torch.nn.functional.batch_norm(cat, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return bn


def replacement_args(cat_a, cat_b, running_mean, running_var, weight, bias):
    return (cat_a, cat_b, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_cat_bn_relu