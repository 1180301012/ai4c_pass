import torch
from pass_dir.dispatch_kernels import dispatch_fn


# =============================================================================
# PATTERN: y * sigmoid(x)
#
# Matches Branch-B of BiSeNetV2 BGA:
#   x = conv2d(in_5) output       [B, C, 16, 16]
#   y = in_2 (detail features)    [B, C, 16, 16]
# =============================================================================
def pattern(x, y):
    t = torch.sigmoid(x)
    return y * t


def replacement_args(x, y):
    # route "sigmoid_mul" → dispatch_fn computes b*sigmoid(a), c=b (dummy)
    return (x, y, y, "sigmoid_mul")


def replacement_func():
    return dispatch_fn