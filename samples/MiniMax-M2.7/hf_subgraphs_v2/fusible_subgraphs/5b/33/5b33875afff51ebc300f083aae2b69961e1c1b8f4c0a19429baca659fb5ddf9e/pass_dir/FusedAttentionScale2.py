import torch


def pattern_scale2(in_0, in_2, in_3):
    """Match attention pattern with scale 2.828..."""
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args_scale2(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "scale2")


# Shared replacement_func - imports from the first pass file
from pass_dir.FusedAttentionScaleAddSoftmaxDropoutMatmul import replacement_func