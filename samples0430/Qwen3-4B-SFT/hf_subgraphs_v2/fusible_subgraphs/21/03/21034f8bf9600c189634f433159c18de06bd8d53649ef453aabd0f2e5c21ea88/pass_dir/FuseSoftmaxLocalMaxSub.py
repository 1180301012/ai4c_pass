import torch
from pass_dir.shared_kernel import fused_max_sub_softmax


def pattern(in_0):
    """
    Matches the classic numerically-stable local softmax subgraph:
      max_over_last_dim -> getitem(values) -> expand_as -> subtract-> softmax
    This pattern mirrors all the fusible_subgraphs in the DANet_R101 models.
    """
    t = torch.max(in_0, -1, keepdim=True)
    tmp_1 = t[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_max_sub_softmax