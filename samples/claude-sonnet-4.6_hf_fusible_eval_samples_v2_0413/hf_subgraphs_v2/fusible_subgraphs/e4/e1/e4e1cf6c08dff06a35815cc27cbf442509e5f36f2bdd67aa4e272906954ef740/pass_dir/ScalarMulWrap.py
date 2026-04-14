import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import dispatch_wrapper


# ---------------------------------------------------------------------------
# Pattern – matches the scalar multiply:  tmp_2 = in_0 * in_2
# Wrapping this in a Triton kernel prevents TorchInductor from emitting a
# GPU kernel for it, eliminating the GPU-sync overhead that would otherwise
# occur before the RMSNorm Python wrapper.
# ---------------------------------------------------------------------------
def pattern(in_0, in_2):
    return in_0 * in_2


def replacement_args(in_0, in_2):
    # Append route string so the shared dispatch_wrapper knows which kernel
    return (in_0, in_2, "scalar_mul")


def replacement_func():
    return dispatch_wrapper