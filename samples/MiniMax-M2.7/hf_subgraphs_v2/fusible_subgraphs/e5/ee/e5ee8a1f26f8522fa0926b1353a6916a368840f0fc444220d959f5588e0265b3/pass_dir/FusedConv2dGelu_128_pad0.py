import torch
from pass_dir.shared_kernel import _fused_conv2d_gelu_dispatch


def pattern(in_0, in_1, in_2):
    """
    Pattern: Conv2D with bias followed by GELU activation (groups=1, padding=0)
    Note: fastvit graphs have groups=1, padding=(0,0)
    Weight shape: [256, 64, 1, 1], in_channels=64, out_channels=256
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.gelu(conv2d, approximate='none')
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    # Route string appended to differentiate passes while sharing replacement_func
    return (in_0, in_1, in_2, 1, 0, 0, "1_pad0")


def replacement_func():
    return _fused_conv2d_gelu_dispatch_wrapper


@torch.fx.wrap
def _fused_conv2d_gelu_dispatch_wrapper(bias, weight, input, groups, padding_h, padding_w, route):
    """Shared dispatch wrapper for all fused Conv2D + GELU patterns"""
    if route == "128_pad1":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    elif route == "256_pad1":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    elif route == "512_pad1":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    elif route == "1024_pad1":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    elif route == "2048_pad1":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    elif route == "1_pad0":
        return _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w)
    else:
        raise ValueError(f"Unknown route: {route}")