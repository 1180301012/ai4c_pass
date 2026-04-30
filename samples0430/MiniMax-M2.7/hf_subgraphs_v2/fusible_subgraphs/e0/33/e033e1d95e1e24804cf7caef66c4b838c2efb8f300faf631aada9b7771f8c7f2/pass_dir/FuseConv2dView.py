import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_0, in_1, in_3):
    """
    Match: conv2d + view fusion pattern.
    This pattern matches the exact FX graph structure.
    """
    # Conv2d with positional args for stride, padding, dilation, groups
    conv_result = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # View to reshape - match the batch dimension from tensor size
    batch_dim = conv_result.shape[0]
    output = conv_result.view(batch_dim, 256, -1)
    return output


def replacement_args(in_0, in_1, in_3):
    # Append route string as last argument for dispatch
    return (in_0, in_1, in_3, "conv2d_view")


def replacement_func():
    return shared_dispatch