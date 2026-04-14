import torch
from pass_dir.shared_dispatch import dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Match both linear AND batch_norm together in one pass."""
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7 = torch.nn.functional.batch_norm(
        in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05
    )
    return (linear, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # dispatch(arg0..arg7, route) → 9 positional args
    # in_0=mean, in_1=var, in_2=beta, in_3=gamma,
    # in_4=lin_bias, in_5=lin_weight, in_6=lin_input, in_7=bn_input
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, "combined")


def replacement_func():
    return dispatch