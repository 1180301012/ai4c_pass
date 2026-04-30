import torch
try:
    from pass_dir.shared_botnet_kernels import replacement_func
except ImportError:
    from shared_botnet_kernels import replacement_func


def pattern(in_0: torch.Tensor, in_1):
    tmp_13 = in_0.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_1
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0: torch.Tensor, in_1):
    return (in_0, in_1, 'softmax_mm_t_256')