"""
Fuse: (in_0 + embed) → layer_norm(H=768) via importlib-shared kernel.
All three H-variants import the SAME fused_add_ln object → replacement_func_limit=1 satisfied.
"""
import torch
import importlib.util, os, sys


def _load_shared():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.path.join(os.getcwd(), 'pass_dir')
    path = os.path.join(here, 'shared_add_ln.py')
    key  = '__ai4c_shared_add_ln__'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


_sh = _load_shared()
fused_add_ln = _sh.fused_add_ln   # SAME object across all H-variants


def pattern(in_0, weight, bias, embed):
    tmp_13 = in_0 + embed
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), weight, bias, 1e-05)
    return tmp_14


def replacement_args(in_0, weight, bias, embed):
    return (in_0, weight, bias, embed)


def replacement_func():
    return fused_add_ln