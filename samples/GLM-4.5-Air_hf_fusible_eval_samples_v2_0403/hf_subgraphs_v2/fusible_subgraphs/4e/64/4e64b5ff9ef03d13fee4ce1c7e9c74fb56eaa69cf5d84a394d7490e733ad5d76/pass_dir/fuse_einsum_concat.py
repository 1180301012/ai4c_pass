import torch
import triton
import triton.language as tl

# Pattern: einsum followed by concatenation - fuse these operations
def pattern(in_2, in_1, in_0):
    einsum_result = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    concat_result = torch.cat([in_0, einsum_result], dim=-1)
    return concat_result

# Extract arguments for replacement
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Optimized implementation that fuses einsum and concatenation
@torch.fx.wrap
def fused_einsum_concat(query, key, energy):
    # Get tensor shapes
    batch, height, width, energy_channels = energy.shape
    _, _, _, key_channels = key.shape
    
    # Total output channels = energy_channels + key_channels
    total_channels = energy_channels + key_channels
    
    # Create output tensor directly with combined shape
    out_shape = (batch, height, width, total_channels)
    out = torch.empty(out_shape, dtype=energy.dtype, device=energy.device)
    
    # Copy energy to the beginning of output
    out[..., :energy_channels] = energy
    
    # Compute matrix multiplication and store in the remaining channels
    einsum_result = torch.matmul(query, key.transpose(-1, -2))
    out[..., energy_channels:] = einsum_result
    
    return out

def replacement_func():
    return fused_einsum_concat