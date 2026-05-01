"""
Pass: FuseBatchNormInference

Matches: batch_norm(x, mean, var, weight, bias, training=False, momentum=0.1, eps=1e-05)
Returns: tmp_11  (SINGLE output)

Replaces inference-mode batch norm with a Triton kernel that computes:
    out = (x - mean) / sqrt(var + 1e-5) * weight + bias
per-channel for NCHW tensors.
"""

import torch
from pass_dir.shared_kernel import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern: 1 op, 1 output (tmp_11)
# ---------------------------------------------------------------------------

def pattern(x, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    result = torch.nn.functional.batch_norm(
        x, bn_running_mean, bn_running_var, bn_weight, bn_bias,
        False, 0.1, 1e-05,
    )
    return result


def replacement_args(x, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    # Route "bn_infer" → _run_bn_infer(x, bn_mean, bn_var, bn_weight, bn_bias)
    return (x, bn_running_mean, bn_running_var, bn_weight, bn_bias, "bn_infer")


def replacement_func():
    # MUST return the SAME fused_dispatch object as all other pass files
    return fused_dispatch