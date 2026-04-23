import torch
from pass_dir.shared_fused_einsum_epilogue import replacement_func


def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_5 = torch.ops.aten.add.out(in_3, einsum, out=in_3)
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_aten_out")