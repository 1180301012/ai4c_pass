import torch
import sys
import os

# Add pass_dir to sys.path so we can import the shared library
pass_dir = os.path.dirname(os.path.abspath(__file__))
if pass_dir not in sys.path:
    sys.path.insert(0, pass_dir)

from fused_qkv_lib import fused_qkv_dispatch


def pattern(in_0, in_1):
    """Pattern matching function for convit_small QKV projection (N_heads=9).
    
    Matches the exact computation:
    1. Linear projection: torch.nn.functional.linear(in_1, in_0, None)
    2. Reshape to (1, 197, 3, 9, 48)
    3. Permute to (3, 1, 9, 197, 48)  i.e. permute(2, 0, 3, 1, 4)
    4. Unbind along dim 0 to get q, k, v
    5. Transpose k to get k_t
    
    Returns all observable outputs: (q, k_t, v)
    """
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function.
    
    Returns (input_tensor, weight_tensor, route_string) where:
    - input_tensor = in_1 (the input to the linear projection)
    - weight_tensor = in_0 (the weight matrix)
    - route_string identifies this pattern variant
    """
    return (in_1, in_0, "nheads9")


def replacement_func():
    """Return the shared dispatch wrapper function.
    
    All pass files must return the SAME function object to satisfy
    the replacement_func_limit constraint.
    """
    return fused_qkv_dispatch