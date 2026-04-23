import torch
import triton
import triton.language as tl
from pass_dir.shared_weighted_sum_sub5 import shared_dispatch


# Match the numerically meaningful tail subgraph.
# We intentionally leave softmax and linspace outside the pattern because
# their FX capture can differ in kwargs/default-arg normalization.
def pattern(tmp_0, tmp_1):
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(tmp_0, tmp_1):
    return (tmp_0, "from_probs")


@triton.jit
def _softmax_weighted_sum_sub5_kernel(
    x_ptr,
    out_ptr,
    stride_x0,
    stride_x1,
    stride_out0,
):
    row = tl.program_id(0)
    row_ptr = x_ptr + row * stride_x0

    p0 = tl.load(row_ptr + 0 * stride_x1).to(tl.float32)
    p1 = tl.load(row_ptr + 1 * stride_x1).to(tl.float32)
    p2 = tl.load(row_ptr + 2 * stride_x1).to(tl.float32)
    p3 = tl.load(row_ptr + 3 * stride_x1).to(tl.float32)
    p4 = tl.load(row_ptr + 4 * stride_x1).to(tl.float32)

    weighted_sum = p0 * 0.0
    weighted_sum = weighted_sum + p1 * 1.0
    weighted_sum = weighted_sum + p2 * 2.0
    weighted_sum = weighted_sum + p3 * 3.0
    weighted_sum = weighted_sum + p4 * 4.0
    out = 5.0 - weighted_sum
    tl.store(out_ptr + row * stride_out0, out)


def replacement_func():
    return shared_dispatch