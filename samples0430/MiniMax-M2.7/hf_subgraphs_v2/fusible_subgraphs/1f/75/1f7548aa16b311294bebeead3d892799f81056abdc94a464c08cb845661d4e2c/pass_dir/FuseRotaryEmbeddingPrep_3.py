"""
Pass for sequence length 3.
Matches the attention mask processing and rotary embedding preparation pattern.
Used by TinyLlama graphs.
"""
import torch
from pass_dir.shared_kernels import fused_rotary_attention_wrapper


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the full computation pattern with arange(3).
    Returns all three outputs that are observable outside the subgraph.
    """
    # === Part 1: Attention mask processing ===
    tmp_2 = in_0.to(device=torch.device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(3, device=torch.device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_3 = None
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_2 = tmp_4 = None
    tmp_6 = torch.arange(3, device=torch.device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_6 = None
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_7 = tmp_8 = None
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = None
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_10 = None
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_5 = None
    tmp_13 = tmp_11 * tmp_12
    tmp_11 = tmp_12 = None
    
    # === Part 2: Rotary embedding preparation for in_1 ===
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_15 = None
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_16 = None
    tmp_18 = tmp_17.to(device=torch.device(type='cuda', index=0))
    tmp_17 = None
    
    # === Part 3: Position ids processing for in_3 ===
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_19 = None
    tmp_21 = tmp_18.float()
    tmp_18 = None
    tmp_22 = tmp_20.float()
    tmp_20 = None
    
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the optimized replacement.
    Route string '3' is appended to distinguish this pass.
    """
    return (in_0, in_1, in_2, in_3, 3)


def replacement_func():
    """Returns the shared routing wrapper function."""
    return fused_rotary_attention_wrapper