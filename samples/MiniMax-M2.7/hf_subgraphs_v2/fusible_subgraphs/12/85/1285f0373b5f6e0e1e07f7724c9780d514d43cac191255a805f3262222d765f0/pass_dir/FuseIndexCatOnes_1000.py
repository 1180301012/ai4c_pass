import torch
from torch import device
import triton
import triton.language as tl

# Pattern for GAE graph
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = tmp_0[slice(None, None, None), in_2]
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([1000, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
    return tmp_9, tmp_11

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "gae")

def replacement_func():
    # Returns the shared dispatcher function (same as other pass files)
    from pass_dir.FuseIndexCatOnes_128 import fused_index_cat_ones_dispatcher
    return fused_index_cat_ones_dispatcher