import torch
import triton
import triton.language as tl
from pass_dir.silu_pool_flat_kernel import fused_silu_pool_flat_dispatch

# Pattern matching function - must exactly mirror model.py operations
def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return (tmp_3,)

# Argument extraction - pass input tensor and route string
def replacement_args(in_0):
    return (in_0, "route_p02")

# Shared replacement_func - same across all passes
def replacement_func():
    return fused_silu_pool_flat_dispatch