import torch
import triton
import triton.language as tl
from pass_dir.relative_pos_kernel import get_relative_pos_table_flat


def pattern(in_0, in_1):
    # Build 14x14 coordinate meshgrid
    tmp_1 = torch.arange(14)
    tmp_2 = torch.arange(14)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    # Stack into [2, 14, 14], then flatten to [2, 196]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    # Broadcast to [14, 14, 2] for pairwise difference
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9
    # Permute -> [196, 196, 2] and make contiguous
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    # Offset row dimension by +13
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 13
    tmp_14 = tmp_13
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14
    # Offset col dimension by +13
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 13
    tmp_17 = tmp_16
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    # Scale row dimension by *27
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 27
    tmp_20 = tmp_19
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20
    # Build final (197, 197) int64 index table
    tmp_22 = torch.zeros(size=(197, 197), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 729
    tmp_22[(slice(0, None, None), 0)] = 730
    tmp_22[(0, 0)] = 731
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def relative_pos_table_197x197(in_0, in_1):
    """
    Replaces the deterministic position-bias table computation.
    in_0 and in_1 are accepted as required by the pass framework but unused
    (the table depends only on grid_size=14, offset=13, multiplier=27).
    """
    return get_relative_pos_table_flat(14, 13, 27)


def replacement_func():
    return relative_pos_table_197x197