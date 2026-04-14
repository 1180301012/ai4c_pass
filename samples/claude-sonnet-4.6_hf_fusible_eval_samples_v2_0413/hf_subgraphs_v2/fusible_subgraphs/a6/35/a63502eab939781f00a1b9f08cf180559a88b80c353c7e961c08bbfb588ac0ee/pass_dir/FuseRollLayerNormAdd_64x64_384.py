import torch
import os
import sys

# Ensure pass_dir itself is importable so we can load the shared kernel module
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)

from shared_kernels import _dispatch_fused_roll_ln_add  # noqa: E402


# ------------------------------------------------------------------ #
#  Pattern: contiguous -> view(-1,64,64,384) -> roll(4,4) ->
#           view(1,4096,384) -> layer_norm -> residual add
# ------------------------------------------------------------------ #

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 4096, 384)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7   # return the tensor directly (not wrapped in a tuple)


def replacement_args(in_0, in_1, in_2, in_3):
    # Make in_3 contiguous (exempt from API restriction) and tag route
    return (in_0, in_1, in_2, in_3.contiguous(), "384")


def replacement_func():
    # Return the SAME shared function object as the 768 pass
    return _dispatch_fused_roll_ln_add