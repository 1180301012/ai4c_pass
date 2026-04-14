import torch
import sys
import os

# Import the shared dispatch wrapper so all passes return the SAME function object
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from fused_add_ln_dispatch import fused_add_layernorm_dispatch  # noqa: E402


def pattern(in_0, in_1, in_2, in_3):
    # FP32 Hubert case: addition operands are reversed (in_3 + in_2)
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "768_rev")


def replacement_func():
    return fused_add_layernorm_dispatch