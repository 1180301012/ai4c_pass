import torch
from pass_dir.shared_kernel import replacement_func

# Pattern matching function - matches (tmp_2, tmp_4) return order
# Using torch.ops.aten.native_layer_norm for ATen-level matching
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.ops.aten.native_layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)[0]
    return (tmp_2, tmp_4)

# Argument extraction function - adds route string for dispatch
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "sum_first")