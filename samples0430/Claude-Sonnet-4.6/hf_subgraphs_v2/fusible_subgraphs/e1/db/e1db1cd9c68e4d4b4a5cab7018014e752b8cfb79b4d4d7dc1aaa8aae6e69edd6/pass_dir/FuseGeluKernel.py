import torch
import triton
import triton.language as tl
import sys, os, importlib.util as _ilu


# ── Load shared kernels module once (same object across all pass files) ──
_SK_KEY = "__pass_shared_kernels"
if _SK_KEY not in sys.modules:
    _spec = _ilu.spec_from_file_location(
        _SK_KEY,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_kernels.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_SK_KEY] = _mod
    _spec.loader.exec_module(_mod)

dispatch_wrapper = sys.modules[_SK_KEY].dispatch_wrapper


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "gelu")


def replacement_func():
    return dispatch_wrapper