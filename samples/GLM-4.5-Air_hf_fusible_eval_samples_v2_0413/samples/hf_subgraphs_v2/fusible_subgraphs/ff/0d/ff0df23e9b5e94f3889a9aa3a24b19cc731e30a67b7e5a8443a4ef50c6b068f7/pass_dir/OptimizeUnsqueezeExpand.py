import torch
import torch.fx
import triton
import triton.language as tl
from torch import fx

# This must be at module level
@fx.wrap
def optimize_unsqueeze_expand(unsqueezed_tensor, target_shape):
    """
    Optimized unsqueeze + expand operation using Triton
    Eliminates the unsqueeze step and directly performs the expansion
    """
    input_shape = unsqueezed_tensor.shape
    
    # For [embed_dim, batch_size, seq_len] -> expand to [1, embed_dim, target_h, target_w]
    if len(input_shape) == 3:
        embed_dim, batch_size, seq_len = input_shape
        target_h, target_w = target_shape[0], target_shape[1]
        
        # Directly compute the expanded result without intermediate unsqueeze
        return triton_direct_expand(unsqueezed_tensor, embed_dim, batch_size, seq_len, target_h, target_w)
    else:
        # Fallback for unsupported shapes
        return unsqueezed_tensor.expand((1, -1, target_shape[0], target_shape[1]))

@triton.jit
def triton_direct_expand_kernel(
    input_ptr,
    output_ptr,
    embed_dim,
    batch_size,
    seq_len,
    target_h,
    target_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Direct expansion: [embed_dim, batch_size, seq_len] -> [1, embed_dim, target_h, target_w]
    Avoids creating intermediate unsqueezed tensor
    """
    # 2D grid for output expansion (target_h, target_w)
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Create workgroup for efficient memory access
    local_h = tl.arange(0, BLOCK_SIZE)
    local_w = tl.arange(0, BLOCK_SIZE)
    mask_h = local_h < target_h
    mask_w = local_w < target_w
    mask = mask_h[:, None] & mask_w[None, :]
    
    # For each embed dimension and batch position
    for embed_idx in range(embed_dim):
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Input offset: [embed_dim, batch_size, seq_len]
                in_offset = embed_idx * (batch_size * seq_len) + batch_idx * seq_len + seq_idx
                
                # Load input value
                input_val = tl.load(input_ptr + in_offset)
                
                # Broadcast to entire target region: [1, embed_dim, target_h, target_w]
                out_offset = embed_idx * (target_h * target_w) + pid_h * target_w + pid_w
                
                # Store broadcasted value
                tl.store(output_ptr + out_offset, input_val, mask)

def triton_direct_expand(input_tensor, embed_dim, batch_size, seq_len, target_h, target_w):
    """
    Direct optimized expansion without intermediate unsqueeze
    """
    # Create output with final shape
    output_shape = (1, embed_dim, target_h, target_w)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use efficient expansion for larger targets
    if target_h * target_w > 1024:
        BLOCK_SIZE = 16
        grid_h = (target_h + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_w = (target_w + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        triton_direct_expand_kernel[(grid_h, grid_w)](
            input_ptr=input_tensor,
            output_ptr=output.view(-1),
            embed_dim=embed_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            target_h=target_h,
            target_w=target_w,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For small expansions, just use PyTorch expand
        tmp_4 = input_tensor.unsqueeze(0)
        return tmp_4.expand((1, -1, target_h, target_w))
    
    return output

def pattern(unsqueezed_tensor, target_shape):
    # Pattern: unsqueeze(0) -> expand((1, -1, target_shape[0], target_shape[1]))
    tmp_4 = unsqueezed_tensor.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, target_shape[0], target_shape[1]))
    return tmp_5

def replacement_args(unsqueezed_tensor, target_shape):
    return (unsqueezed_tensor, target_shape)

def replacement_func():
    return optimize_unsqueeze_expand