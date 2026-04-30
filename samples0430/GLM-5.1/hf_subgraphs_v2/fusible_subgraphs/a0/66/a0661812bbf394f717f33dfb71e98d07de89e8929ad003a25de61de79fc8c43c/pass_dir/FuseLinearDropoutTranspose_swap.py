import torch
import importlib.util
import os
import sys

# Load the shared kernel module
_fused_kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_kernel.py")
_spec = importlib.util.spec_from_file_location("fused_kernel", _fused_kernel_path)
_fused_kernel_module = importlib.util.module_from_spec(_spec)
sys.modules["fused_kernel"] = _fused_kernel_module
_spec.loader.exec_module(_fused_kernel_module)

fused_linear_dropout_transpose_dispatch = _fused_kernel_module.fused_linear_dropout_transpose_dispatch


# Pattern matching function - matches linear -> dropout(training=False) -> transpose
# Returns (transpose_result, dropout_result) order
def pattern(in_0, in_1, in_2, p):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    dropout = torch.nn.functional.dropout(linear, p, False, False)
    transposed = dropout.transpose(1, 2)
    return (transposed, dropout)


# Argument extraction function - includes route string for dispatch
def replacement_args(in_0, in_1, in_2, p):
    return (in_0, in_1, in_2, "t4_d3")


# Replacement function - returns the shared dispatch wrapper
def replacement_func():
    return fused_linear_dropout_transpose_dispatch