import torch
import triton
import triton.language as tl

def pattern(dummy_input, position_indices):
    # Position index computation
    ones = torch.ones((1, 15), dtype=torch.int64)
    cumsum_ones = torch.cumsum(ones, dim=1)
    result = cumsum_ones - ones
    result += 2
    
    return result

def replacement_args(dummy_input, position_indices):
    return (dummy_input, position_indices)

# Removed unused kernel functions for simplicity

@torch.fx.wrap
def optimized_position_indices(dummy_input, position_indices):
    # Optimized position index computation
    num_positions = 15
    
    result = torch.empty((1, 15), dtype=torch.int64, device=position_indices.device)
    
    # Simple optimized computation that matches the pattern exactly
    positions = torch.arange(2, 17, dtype=torch.int64, device=position_indices.device)
    result[0, :] = positions
    
    return result

def replacement_func():
    return optimized_position_indices