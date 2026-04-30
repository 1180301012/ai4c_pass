import torch
from pass_dir.shared_kernels import dispatch_matmul


def pattern(in_0, in_1):
    result = in_1 @ in_0
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1, "gemm_op")


def replacement_func():
    return dispatch_matmul