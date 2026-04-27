import torch

# Re-use the Triton kernels and wrapper that are already compiled in the
# primary pass file.  Importing them here avoids duplicating the kernel source
# and ensures Triton only compiles one instance per (dtype, BLOCK_HW) pair.
from pass_dir.FuseDropoutScaleAddBatchNorm import triton_scale_add   # noqa: F401


# ---------------------------------------------------------------------------
# Pattern: dropout(p=0, training=False) + layer-scale multiply + residual add
#
# Why this pass exists:
#   When dropout IS present as a graph node (even though it is a mathematical
#   no-op), it still costs one FX-interpreter node dispatch (~50 µs on A30).
#   By including dropout in the pattern we fold it into the single Triton
#   kernel call and remove that overhead from the compiled graph.
#
# If dropout has already been folded away by the framework (i.e. the target
# graph contains no dropout node), this pattern simply finds no match and the
# fallback pass (FuseDropoutScaleAddBatchNorm) fires instead.
# ---------------------------------------------------------------------------
def pattern(conv_out, gamma, residual):
    dropped = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scaled  = dropped * gamma
    pre_bn  = residual + scaled
    return pre_bn


def replacement_args(conv_out, gamma, residual):
    # conv_out here is the RAW conv2d output (before dropout).
    # Since dropout(p=0, train=False) is identity, the kernel produces
    # the same result as residual + conv_out * gamma.
    return (conv_out, gamma, residual)


def replacement_func():
    return triton_scale_add