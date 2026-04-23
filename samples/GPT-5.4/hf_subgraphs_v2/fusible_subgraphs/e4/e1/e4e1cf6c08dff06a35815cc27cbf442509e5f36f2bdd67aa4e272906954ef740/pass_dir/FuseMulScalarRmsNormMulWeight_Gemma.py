import torch
import triton
import triton.language as tl
from pass_dir.gemma_rmsnorm_shared import gemma_rmsnorm_dispatch


def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module')
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return (tmp_2, tmp_13)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'full')


def replacement_func():
    return gemma_rmsnorm_dispatch