import torch
from pass_dir.linear_kernel_shared import fused_linear_wrapper


def pattern(in_0, in_1, in_2):
    """
    Matches:
        tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
        linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    Found in bigbird-roberta-base (bfloat16 and float16 variants).
    dropout with training=False is a no-op; fuse into a single GEMM+bias kernel.
    """
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    # (bias, weight, input) order matches fused_linear_wrapper signature
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_wrapper