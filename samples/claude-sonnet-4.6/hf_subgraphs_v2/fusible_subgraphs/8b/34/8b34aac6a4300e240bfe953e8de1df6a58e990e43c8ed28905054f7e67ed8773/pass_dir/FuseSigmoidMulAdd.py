import torch
from pass_dir.dispatch_kernels import dispatch_fn


# =============================================================================
# PATTERN: sigmoid(x) * y + z
#
# In the BiSeNetV2 BGA graph this matches:
#   x = interpolate(in_4)     [B, C, 64, 64]   – post-upsample attention
#   y = in_3                  [B, C, 64, 64]   – context feature
#   z = interpolate(in_2 *    [B, C, 64, 64]   – upsampled detail branch
#          sigmoid(conv2d…))
#
# Fusing these three ops (sigmoid, mul, add) into one kernel saves
# 5 memory round-trips vs executing them separately.
# =============================================================================
def pattern(x, y, z):
    t   = torch.sigmoid(x)
    mul = y * t
    return mul + z


def replacement_args(x, y, z):
    # route "sigmoid_mul_add" → dispatch_fn computes b*sigmoid(a)+c
    return (x, y, z, "sigmoid_mul_add")


def replacement_func():
    return dispatch_fn