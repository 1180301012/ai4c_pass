import torch
import importlib.util as imp
import os
import sys

# Load shared kernel module (ensuring single instance via sys.modules)
def _ensure_shared_kernels():
    module_name = '_kernels_shared_instance'
    if module_name not in sys.modules:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(this_dir, '_kernels.py')
        spec = imp.spec_from_file_location(module_name, kernel_path)
        module = imp.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]

_shared = _ensure_shared_kernels()


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(64, -1, 16, 16)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _shared.fused_linear_permute_reshape_interpolate