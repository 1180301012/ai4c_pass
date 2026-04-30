import torch
from torch import device
from pass_dir.shared_fused_graph_ops import graph_fused_dispatch


def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    torch._check_is_size(tmp_2)
    tmp_4 = tmp_2 >= 0
    torch.ops.aten._assert_scalar.default(tmp_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
    tmp_6 = tmp_2 <= 128
    torch.ops.aten._assert_scalar.default(tmp_6, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'")
    torch._check_is_size(tmp_2)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
    return (tmp_9, tmp_11)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "full")


def replacement_func():
    return graph_fused_dispatch