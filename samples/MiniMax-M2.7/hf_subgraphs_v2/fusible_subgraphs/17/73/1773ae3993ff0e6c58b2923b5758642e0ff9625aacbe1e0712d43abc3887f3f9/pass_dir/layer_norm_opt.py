import torch


# ============================================================================
# LayerNorm Pattern for C=384 (yolos-small, float32)
# ============================================================================
def pattern(in_4, in_1, in_0):
    """Pattern matcher for layer_norm with C=384"""
    tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
    return tmp_3


def replacement_args(in_4, in_1, in_0):
    """Extract arguments for layer_norm replacement"""
    return (in_4, in_1, in_0)


# ============================================================================
# SHARED Replacement Function (uses shared layer_norm impl)
# ============================================================================
def replacement_func():
    """Returns the layer_norm implementation function"""
    from pass_dir.shared_kernels import _layer_norm_impl
    return _layer_norm_impl