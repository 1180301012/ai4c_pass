import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    """
    Pattern matching SE attention: sigmoid -> view -> expand_as -> multiply
    """
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@torch.fx.wrap
def fused_se_attention_simple(in_1, in_2):
    """
    Simplified SE attention - just use broadcasting.
    Avoids the expand_as operation.
    """
    # Directly reshape and multiply, relying on broadcasting
    return in_1 * in_2.sigmoid().view(1, -1, 1, 1)


def replacement_func():
    return fused_se_attention_simple