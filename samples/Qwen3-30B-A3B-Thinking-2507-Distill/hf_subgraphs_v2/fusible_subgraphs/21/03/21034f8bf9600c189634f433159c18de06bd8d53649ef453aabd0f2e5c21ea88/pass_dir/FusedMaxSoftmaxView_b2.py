import torch
import triton
import triton.language as tl
from pass_dir.fused_softmax_shared import fused_max_softmax_wrapper

_DTYPE_FP16 = tl.float16
_DTYPE_BF16 = tl.bfloat16
_DTYPE_FP32 = tl.float32


def pattern(in_0, in_1):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(2, 512, -1)
    return (tmp_4, tmp_5)


def replacement_args(in_0, in_1):
    B, C, _ = in_1.shape
    dtype = _DTYPE_FP16 if in_0.dtype == torch.float16 else (_DTYPE_BF16 if in_0.dtype == torch.bfloat16 else _DTYPE_FP32)
    return (in_0, in_1, B, C, dtype)


def replacement_func():
    return fused_max_softmax_wrapper