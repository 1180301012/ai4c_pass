import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import _run_fused_conv1x1_bilinear


def pattern(in_10, in_8, in_7):
    """
    ATen: aten.convolution.default + aten.upsample_bilinear2d.vec
    """
    conv2d = torch.ops.aten.convolution.default(
        in_10, in_8, in_7,
        [1, 1], [0, 0], [1, 1], False, [0, 0], 1
    )
    tmp_11 = torch.ops.aten.upsample_bilinear2d.vec(conv2d, [512, 512], False, None)
    return tmp_11


def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7, "route_vec")


@torch.fx.wrap
def dispatch_wrapper(in_10, in_8, in_7, route):
    if route == "route_default":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    elif route == "route_vec":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    elif route == "route_py":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)


def replacement_func():
    return dispatch_wrapper