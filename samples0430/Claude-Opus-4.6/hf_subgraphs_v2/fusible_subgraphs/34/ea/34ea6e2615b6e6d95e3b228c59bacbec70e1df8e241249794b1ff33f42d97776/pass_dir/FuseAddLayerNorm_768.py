import torch
import triton
import triton.language as tl
import sys
import os
import importlib.util


def _get_shared():
    key = '__pass_shared_layernorm_v1__'
    if key in sys.modules:
        return sys.modules[key]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(key, os.path.join(dir_path, '_shared.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[key] = mod
    return mod


_shared = _get_shared()


def pattern(conv_out, in_4):
    tmp_5 = conv_out + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


def replacement_args(conv_out, in_4):
    return (conv_out, in_4, 0)


def replacement_func():
    return _shared.custom_dispatch