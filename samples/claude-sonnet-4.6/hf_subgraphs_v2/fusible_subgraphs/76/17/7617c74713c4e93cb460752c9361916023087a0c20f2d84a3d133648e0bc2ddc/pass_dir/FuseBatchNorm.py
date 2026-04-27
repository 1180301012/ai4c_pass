"""
Pass: FuseBatchNorm
Matches: batch_norm inference call on a [B,C] input.
Replaces with a Triton kernel that computes (x-mean)/sqrt(var+eps)*w+b.
Returns a SINGLE output (the BN result), keeping the assertion count correct.
"""
import torch
from pass_dir.shared_ops import shared_dispatch  # same object as FuseAddMean → limit=1 satisfied


# ---------------------------------------------------------------------------
# Pattern: batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-5)
# The argument ORDER must mirror model.py:
#   batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
#   = batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
# ---------------------------------------------------------------------------

def pattern(running_mean, running_var, bias, weight, x):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)


def replacement_args(running_mean, running_var, bias, weight, x):
    # Append route string; shared_dispatch will call _run_batch_norm
    return (running_mean, running_var, bias, weight, x, "batch_norm")


def replacement_func():
    return shared_dispatch