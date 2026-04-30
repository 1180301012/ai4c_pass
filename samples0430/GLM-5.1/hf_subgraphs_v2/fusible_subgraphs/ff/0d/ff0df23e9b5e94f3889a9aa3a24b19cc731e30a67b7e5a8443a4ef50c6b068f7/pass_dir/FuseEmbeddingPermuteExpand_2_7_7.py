import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import dispatch_wrapper


def pattern(weight, indices_cuda):
    emb = torch.nn.functional.embedding(indices_cuda, weight, None, None, 2.0, False, False)
    perm = emb.permute([2, 0, 1])
    unsq = perm.unsqueeze(0)
    exp = unsq.expand((2, -1, 7, 7))
    cont = exp.contiguous()
    return (cont,)


def replacement_args(weight, indices_cuda):
    return (weight, indices_cuda, "route_c")


def replacement_func():
    return dispatch_wrapper