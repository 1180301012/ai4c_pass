import torch
from pass_dir.linear_kernel_shared import fused_linear_wrapper


def pattern(in_0, in_1, in_2):
    """
    Matches:
        to    = in_2.to(torch.float16)
        linear = torch.nn.functional.linear(to, in_1, in_0)
    Found in RECT_L (float16 variant).
    dropout(p=0.0, training=False) is eliminated by the tracer; only the
    .to() cast + linear remain.  Fuse into a single GEMM+bias kernel.
    """
    to = in_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    # (bias, weight, input) order matches fused_linear_wrapper signature
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_wrapper