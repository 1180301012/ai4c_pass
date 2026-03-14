import torch

def pattern(tmp_10):
    """Match chunk operation with immediate indexing"""
    tmp_15 = tmp_10.chunk(2, dim=1)
    tmp_16 = tmp_15[0]
    tmp_17 = tmp_15[1]
    return tmp_16, tmp_17

def replacement_args(tmp_10):
    return (tmp_10,)

def fused_chunk_slicing(tmp_10):
    """Direct slicing without intermediate chunk tuple creation"""
    # Instead of creating chunk tuple and then indexing, use direct slicing
    # For input_tensor [batch, channels, height, width] with channels divisible by 2:
    # chunk(2, dim=1) followed by [0] and [1] is equivalent to:
    # input_tensor[:, :channels//2, :, :] and input_tensor[:, channels//2:, :, :]
    
    channels = input_tensor.shape[1]
    half_channels = channels // 2
    
    # Direct slicing is more efficient than chunk + indexing
    part1 = input_tensor[:, :half_channels, :, :]
    part2 = input_tensor[:, half_channels:, :, :]
    
    return part1, part2

def replacement_func():
    return fused_chunk_slicing