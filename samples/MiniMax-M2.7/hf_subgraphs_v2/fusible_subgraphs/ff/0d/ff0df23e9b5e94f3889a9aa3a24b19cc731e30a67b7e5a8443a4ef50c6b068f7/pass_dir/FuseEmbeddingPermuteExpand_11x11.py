"""
Pass to fuse embedding lookup + permute + unsqueeze + expand + contiguous operations.
Pattern for graphs with indices shape [11, 11] and expand (1, -1, 11, 11).
"""
import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern for [11, 11] indices shape:
    1. Move indices to GPU
    2. Embedding lookup
    3. Permute [2, 0, 1]
    4. Unsqueeze(0)
    5. Expand to (1, -1, 11, 11)
    6. Contiguous
    """
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 11, 11))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel.
    """
    head_dim = in_0.shape[1]
    expand_batch = 1
    return (in_0, in_1, head_dim, expand_batch)


def replacement_func():
    """
    Returns the replacement function for the fused operation.
    """
    from pass_dir.shared_kernel import fused_embedding_dispatch
    return fused_embedding_dispatch