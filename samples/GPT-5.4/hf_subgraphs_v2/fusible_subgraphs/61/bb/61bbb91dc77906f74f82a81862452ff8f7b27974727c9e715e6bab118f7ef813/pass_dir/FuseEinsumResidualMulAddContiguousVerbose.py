import torch
from pass_dir.shared_fused_einsum_epilogue import replacement_func


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_0 = in_0
    tmp_1 = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += tmp_1
    tmp_2 = in_3
    tmp_3 = tmp_2 * tmp_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_verbose")