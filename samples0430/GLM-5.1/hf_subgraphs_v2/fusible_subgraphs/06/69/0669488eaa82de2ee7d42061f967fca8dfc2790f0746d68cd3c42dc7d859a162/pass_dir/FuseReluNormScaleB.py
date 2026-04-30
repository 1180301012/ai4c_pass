import torch
import sys
import os

# Add pass_dir to sys.path for shared kernel import
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)

from _shared_kernel import replacement_func

# Determine which norm function to use based on what's available
try:
    _torch_norm = torch.functional.norm
except (AttributeError, ModuleNotFoundError):
    _torch_norm = torch.linalg.norm


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = _torch_norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_scale_b")