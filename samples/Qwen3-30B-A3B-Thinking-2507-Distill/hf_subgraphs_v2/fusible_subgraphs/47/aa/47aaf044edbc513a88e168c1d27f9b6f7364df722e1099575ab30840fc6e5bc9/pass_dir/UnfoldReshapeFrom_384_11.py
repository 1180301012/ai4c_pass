import torch
import triton
import triton.language as tl
from pass_dir.shared_unfold_kernel import triton_unfold_dispatch


# Full chain with aten.im2col for YituTech [1, 384, 11] model
def pattern(tmp_1):
    tmp_2 = torch.ops.aten.im2col.default(tmp_1, [9, 1], [1, 1], [4, 0], [1, 1])
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(tmp_1):
    return (tmp_1, 'im2col_384_11')


def replacement_func():
    return triton_unfold_dispatch