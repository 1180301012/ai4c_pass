import torch


def pattern(tmp_3):
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(tmp_3):
    return (tmp_3,)


@torch.fx.wrap
def identity_reshape(x):
    # The reshape/permute/contiguous/permute/reshape chain is mathematically
    # identity (same values, same shape, same memory layout as layer_norm output),
    # so we just return the input directly, eliminating 2 unnecessary memory copies
    # (contiguous() after permute and implicit contiguous in final reshape)
    return x


def replacement_func():
    return identity_reshape