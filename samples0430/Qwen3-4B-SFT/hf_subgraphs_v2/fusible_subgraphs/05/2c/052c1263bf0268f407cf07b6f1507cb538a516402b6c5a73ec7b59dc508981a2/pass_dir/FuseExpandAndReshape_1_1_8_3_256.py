import torch
from pass_dir.shared_kernels import dispatch


# ---------------------------------------------------------------------------
# Pattern: matches the expand + reshape chain that produces tmp_9.
#   tmp_7 = tmp_6[..., None, None]      (unsqueeze dims 1 & 2)
#   tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
#   tmp_9 = tmp_8.reshape(1, 8, 3, 256)  (contiguous copy to new shape)
#
# tmp_9 is OBSERVABLE (it's in the model's return tuple).
# tmp_7 and tmp_8 are internal nodes (not returned by the model).
# ---------------------------------------------------------------------------
def pattern(tmp_6):
    tmp_7 = tmp_6[
        slice(None, None, None),
        slice(None, None, None),
        None,
        slice(None, None, None),
        slice(None, None, None),
    ]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    return tmp_9


def replacement_args(tmp_6):
    # Append route tag so dispatch() calls the right kernel
    return (tmp_6, "expand")


def replacement_func():
    return dispatch