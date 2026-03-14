import torch


def pattern(in_0, in_1):
    # Use slicing instead of split for better FX compatibility
    added = in_1 + in_0
    first_part = added[:, :1, :]
    second_part = added[:, 1:, :]
    permuted = second_part.permute(0, 2, 1)
    output = permuted.view(1, 384, 24, 24)
    return first_part, output


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def fused_kernel_wrapper_384(in_0, in_1):
    out_0 = in_0[:, :1, :] + in_1[:, :1, :]
    combined = in_0 + in_1
    split_part = combined[:, 1:, :]
    permuted = split_part.permute(0, 2, 1)
    out_1 = permuted.view(1, 384, 24, 24)
    return out_0, out_1


def replacement_func():
    return fused_kernel_wrapper_384