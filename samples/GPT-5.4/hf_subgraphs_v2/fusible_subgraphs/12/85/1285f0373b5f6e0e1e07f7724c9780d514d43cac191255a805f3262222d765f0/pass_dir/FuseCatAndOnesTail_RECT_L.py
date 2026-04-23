import torch
import triton
import triton.language as tl

from pass_dir.shared_cat_ones_impl import dispatch_cat_ones


def pattern(tmp_1, in_1):
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    tmp_3 = torch._check_is_size(tmp_2)
    tmp_4 = tmp_2 >= 0
    tmp_5 = torch.ops.aten._assert_scalar.default(tmp_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
    tmp_6 = tmp_2 <= 128
    tmp_7 = torch.ops.aten._assert_scalar.default(tmp_6, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'")
    tmp_8 = torch._check_is_size(tmp_2)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=torch.device(type='cuda'))
    return (tmp_9, tmp_11)


def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1, 128)


def replacement_func():
    return dispatch_cat_ones