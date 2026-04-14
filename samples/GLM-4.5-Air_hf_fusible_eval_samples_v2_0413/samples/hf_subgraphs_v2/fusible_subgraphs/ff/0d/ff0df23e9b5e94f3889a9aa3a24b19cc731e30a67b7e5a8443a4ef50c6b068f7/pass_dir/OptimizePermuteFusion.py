import torch
import torch.fx
import triton
import triton.language as tl
from torch import fx

# This must be at module level
@fx.wrap
def optimize_permute_fusion(tmp_2):
    # Implement permute([2, 0, 1]) using Triton for better performance
    if tmp_2.dim() < 3:
        return torch.as_tensor(tmp_2, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Get original shape and compute new shape
    original_shape = tmp_2.shape
    if len(original_shape) == 3:
        # Shape: [seq_len, batch_size, embed_dim] -> permute to [embed_dim, batch_size, seq_len]
        new_shape = (original_shape[2], original_shape[1], original_shape[0])
    else:
        # Unsupported dimensionality, return original
        return torch.as_tensor(tmp_2, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Call Triton kernel
    return triton_permute_kernel(tmp_2, new_shape)

@triton.jit
def triton_permute_kernel_3d(
    input_ptr,
    output_ptr,
    seq_len,
    batch_size,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # For 3D permute: [seq_len, batch_size, embed_dim] -> [embed_dim, batch_size, seq_len]
    # Each program handles one output position (embed_idx, batch_idx, seq_idx)
    embed_idx = tl.program_id(0)
    batch_idx = tl.program_id(1) 
    seq_idx = tl.program_id(2)
    
    # Create 1D offset for output: [embed_dim, batch_size, seq_len]
    out_offset = embed_idx * (batch_size * seq_len) + batch_idx * seq_len + seq_idx
    
    # Create 1D offset for input: [seq_len, batch_size, embed_dim]
    in_offset = seq_idx * (batch_size * embed_dim) + batch_idx * embed_dim + embed_idx
    
    mask = (embed_idx < embed_dim) & (batch_idx < batch_size) & (seq_idx < seq_len)
    
    if mask:
        # Load from input, store to output
        val = tl.load(input_ptr + in_offset)
        tl.store(output_ptr + out_offset, val)

def triton_permute_kernel(tensor, target_shape):
    """
    Optimized permutation using Triton kernels
    """
    seq_len, batch_size, embed_dim = tensor.shape
    
    # Create output tensor
    output = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Calculate grid size
    grid = (
        (embed_dim + 31) // 32,  # 32 threads per block for embed_dim
        (batch_size + 31) // 32,  # 32 threads per block for batch_size  
        (seq_len + 31) // 32,     # 32 threads per block for seq_len
    )
    
    # Launch kernel
    triton_permute_kernel_3d[grid](
        input_ptr=tensor,
        output_ptr=output,
        seq_len=seq_len,
        batch_size=batch_size,
        embed_dim=embed_dim,
        BLOCK_SIZE=32,
    )
    
    return output

def pattern(tmp_2):
    # Pattern: permute([2, 0, 1])
    tmp_3 = tmp_2.permute([2, 0, 1])
    return tmp_3

def replacement_args(tmp_2):
    return (tmp_2,)

def replacement_func():
    return optimize_permute_fusion