"""
Test pass: full fusion of add+mean+dropout+dropout+batch_norm.

Tries to match the ENTIRE model graph with a single replacement that
computes add+mean+BN in one Triton kernel, returning (tmp_8, tmp_7).

If this crashes, the existing FuseAddMeanDropoutBatchNorm + FuseBatchNorm
passes are the fallback (loaded from the sorted JSON).
If this matches, it gives a tighter fusion with fewer kernel launches.
"""
import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Full computation: add + spatial_mean + dropout(p=0.0)x2 + BN(inference).
    Returns (tmp_8, tmp_7) = (bn_output, mean_output).
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_8, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # route "full_fusion": pass all 6 tensors
    return (in_0, in_1, in_2, in_3, in_4, in_5, "full_fusion")


def replacement_func():
    return dispatch_wrapper