import torch
try:
    from pass_dir.shared_botnet_kernels import replacement_func
except ImportError:
    from shared_botnet_kernels import replacement_func


def pattern(in_0: torch.Tensor, in_1, in_2):
    tmp_7 = in_1.reshape(4, 8, 1, 8, 8)
    tmp_8 = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in_0 + tmp_11
    return tmp_12


def replacement_args(in_0: torch.Tensor, in_1, in_2):
    return (in_0, in_1, in_2, 'shiftadd8')