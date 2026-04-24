import torch
import triton
import triton.language as tl
from pass_dir.shared_unfold_kernel import triton_unfold_dispatch


# Full chain with aten.im2col (decomposed form of unfold)
# tmp_1: [1, 16, 45, 1] (contiguous, after unsqueeze of [1, 16, 45])
def pattern(tmp_1):
    tmp_2 = torch.ops.aten.im2col.default(tmp_1, [9, 1], [1, 1], [4, 0], [1, 1])
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(tmp_1):
    return (tmp_1, 'unfold_16_45')


def replacement_func():
    return triton_unfold_dispatch