import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Try to match just the embedding + type conversion pattern
    emb_out = torch.nn.functional.embedding(in_1, in_0, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    result = emb_out.to(dtype=torch.float32)
    return result

def replacement_args(in_0, in_1):
    position_ids_shape = in_1.shape
    return (in_0, in_1, position_ids_shape)

@triton.jit
def fused_embedding_kernel(
    weight_ptr,
    position_ids_ptr,
    output_ptr,
    n_positions,
    n_embed_dim,
    n_arange,
    offset_add: tl.constexpr,
    offset_sub: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one position and one arange offset
    pid_pos = tl.program_id(0)
    pid_arange = tl.program_id(1)
    pid_embed = tl.program_id(2)
    
    # Boundary checks
    if pid_pos >= n_positions:
        return
    if pid_arange >= n_arange:
        return
    if pid_embed >= n_embed_dim:
        return
    
    # Load position value for this row (position_ids is [n_positions, 1])
    pos_val = tl.load(position_ids_ptr + pid_pos)
    
    # Compute the embedding index for this arange offset
    # tmp_3 = in_1 - tmp_2 = pos_val - arange_offset  
    # tmp_4 = tmp_3 + 2048 = pos_val - arange_offset + 2048
    # tmp_5 = tmp_4 - 1 = pos_val - arange_offset + 2047
    embed_idx = pos_val - pid_arange + offset_add - offset_sub
    
    # Bounds checking for embedding index
    valid_idx = (embed_idx >= 0) & (embed_idx < 4095)  # weight shape[0] = 4095
    
    if valid_idx:
        # Calculate output position: [n_positions, n_arange, n_embed_dim]
        out_offset = (pid_pos * n_arange + pid_arange) * n_embed_dim + pid_embed
        
        # Calculate weight offset: embed_idx * embed_dim + embed_pos
        weight_offset = embed_idx * n_embed_dim + pid_embed
        
        # Load weight and store to output
        weight_val = tl.load(weight_ptr + weight_offset)
        tl.store(output_ptr + out_offset, weight_val)

@torch.fx.wrap
def fused_embedding_lookup(weight, position_ids, position_ids_shape):
    n_positions = position_ids_shape[0]  # 128
    n_embed_dim = weight.shape[1]        
    arange_size = n_positions            # Match arange size to input positions
    
    # Output shape should be [n_positions, arange_size, n_embed_dim]
    output = torch.empty((n_positions, arange_size, n_embed_dim), dtype=torch.float32, device=weight.device)
    
    BLOCK_SIZE = 1024  # Block size for embedding dimension
    grid = (
        n_positions,           # Each program handles one position
        arange_size,           # Each program handles one arange offset
        (n_embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE  # Embed dimension in blocks
    )
    
    fused_embedding_kernel[grid](
        weight_ptr=weight,
        position_ids_ptr=position_ids,
        output_ptr=output,
        n_positions=n_positions,
        n_embed_dim=n_embed_dim,
        n_arange=arange_size,
        offset_add=2048,
        offset_sub=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_embedding_lookup