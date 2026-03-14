import torch

def pattern(tensor_for_chunking):
    """Match any tensor followed by chunk + indexing"""
    chunk_result = tensor_for_chunking.chunk(2, dim=1)
    first_part = chunk_result[0]
    second_part = chunk_result[1]
    return first_part, second_part

def replacement_args(tensor_for_chunking):
    return (tensor_for_chunking,)

def optimized_chunk_splitting(tensor_for_chunking):
    """Direct slicing without intermediate chunk tuple creation"""
    # Use direct slicing for better performance
    channels = tensor_for_chunking.shape[1]
    half_channels = channels // 2
    
    part1 = tensor_for_chunking[:, :half_channels, :, :]
    part2 = tensor_for_chunking[:, half_channels:, :, :]
    
    return part1, part2

def replacement_func():
    return optimized_chunk_splitting