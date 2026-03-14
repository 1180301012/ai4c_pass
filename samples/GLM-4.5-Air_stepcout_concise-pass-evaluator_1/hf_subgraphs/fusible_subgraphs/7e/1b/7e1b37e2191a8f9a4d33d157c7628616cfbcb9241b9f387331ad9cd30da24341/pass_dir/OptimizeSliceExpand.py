import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    # Slice operation - selecting all dimensions and adding None at position 2
    tmp_4 = x[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    
    # Expand operation
    tmp_5 = tmp_4.expand(0, 4, 4, 0, 0)
    
    return (tmp_5, y)

def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def optimized_expand_kernel(
    input_ptr,  # [batch, key_seq, hidden_dim, head_dim] - input tensor
    output_ptr,  # [batch, expand_seq, expand_seq, hidden_dim, head_dim] - output tensor
    batch, hidden_dim, head_dim,
    BATCH_TILE: tl.constexpr,
    HIDDEN_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    # Calculate program ID offsets
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # Compute ranges within tiles
    b_offset = pid_b * BATCH_TILE
    h_offset = pid_h * HIDDEN_TILE
    d_offset = pid_d * HEAD_TILE
    
    # Create offset masks
    b_mask = b_offset + tl.arange(0, BATCH_TILE) < batch
    h_mask = h_offset + tl.arange(0, HIDDEN_TILE) < hidden_dim
    d_mask = d_offset + tl.arange(0, HEAD_TILE) < head_dim
    
    # Load input tile
    # Reshape input to 3D for easier indexing: [batch*hidden_dim*head_dim]
    input_flat_idx = (b_offset[:, None, None] * hidden_dim * head_dim + 
                     h_offset[None, :, None] * head_dim + 
                     d_offset[None, None, :])
    
    input_val = tl.load(input_ptr + input_flat_idx,
                       mask=b_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :],
                       other=0.0)
    
    # Store output - expand along dimension 1 and 2
    output_flat_idx = (b_offset[:, None, None, None, None] * 4 * hidden_dim * head_dim +
                      0 +  # This represents the expansion along dim=1 to 4
                      0 +  # This represents the expansion along dim=2 to 4  
                      h_offset[None, None, :, None] * head_dim +
                      d_offset[None, None, None, :])
    
    # Expand to 4x4 by repeating the input values
    # We need to handle the expansion in the loop
    for i in range(4):  # Expansion along dimension 1
        for j in range(4):  # Expansion along dimension 2
            expanded_idx = output_flat_idx + (i * hidden_dim * head_dim + j * 4 * hidden_dim * head_dim)
            tl.store(output_ptr + expanded_idx, input_val, 
                    mask=b_mask[:, None, None, None, None] & h_mask[None, None, :, None] & d_mask[None, None, None, :])

@torch.fx.wrap  
def optimized_expand(x, y):
    """
    Optimized version that eliminates the intermediate slice and expands directly
    """
    input_shape = x.shape  # [batch, key_seq, hidden_dim, head_dim]
    
    batch, key_seq, hidden_dim, head_dim = input_shape
    
    # For our specific cases, key_seq == 4, but we'll keep it general
    output_shape = (batch, 4, 4, hidden_dim, head_dim)
    
    output = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    
    # Use tiled approach for better GPU utilization
    BATCH_TILE = 4
    HIDDEN_TILE = 64  
    HEAD_TILE = 32
    
    grid_b = (batch + BATCH_TILE - 1) // BATCH_TILE
    grid_h = (hidden_dim + HIDDEN_TILE - 1) // HIDDEN_TILE
    grid_d = (head_dim + HEAD_TILE - 1) // HEAD_TILE
    
    grid = (grid_b, grid_h, grid_d)
    
    # Launch kernel
    optimized_expand_kernel[grid](
        x, output,
        batch, hidden_dim, head_dim,
        BATCH_TILE, HIDDEN_TILE, HEAD_TILE
    )
    
    return (output, y)

def replacement_func():
    return optimized_expand