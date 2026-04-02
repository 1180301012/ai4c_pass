import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_0[None, None, :] — in_0 is a model placeholder, tmp_7 is in
# the model's return. Placeholder->placeholder match is guaranteed by the
# framework. This is a view (no computation), replacement is equivalent.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def gamma_view_wrapper(in_0):
    # Equivalent view: [2, 128] -> [1, 1, 2, 128]
    return in_0[(None, None, slice(None, None, None))]


def replacement_func():
    return gamma_view_wrapper