import torch
import torch.fx
import triton
import triton.language as tl
from torch import fx

# This must be at module level
@fx.wrap
def full_tensor_sequence_fusion(tmp_2, target_shape):
    """
    Fusion of: permute([2, 0, 1]) -> unsqueeze(0) -> expand((1, -1, target_shape[0], target_shape[1])) -> contiguous()
    This eliminates intermediate tensors and performs direct computation
    """
    if tmp_2.dim() < 3:
        return torch.as_tensor(tmp_2, dtype=tmp_2.dtype, device=tmp_2.device)
    
    original_shape = tmp_2.shape
    if len(original_shape) != 3:
        # Unsupported dimensionality, fall back to PyTorch implementation
        tmp_3 = tmp_2.permute([2, 0, 1])
        tmp_4 = tmp_3.unsqueeze(0) 
        tmp_5 = tmp_4.expand((1, -1, target_shape[0], target_shape[1]))
        return tmp_5.contiguous()
    
    seq_len, batch_size, embed_dim = original_shape
    target_w, target_h = target_shape[1], target_shape[0]
    
    # Call Triton kernel to perform the entire sequence
    return triton_full_sequence_kernel(tmp_2, seq_len, batch_size, embed_dim, target_w, target_h)

@triton.jit
def triton_full_sequence_kernel_3d(
    input_ptr,
    output_ptr,
    seq_len,
    batch_size,
    embed_dim,
    target_w,
    target_h,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses: [seq_len, batch_size, embed_dim] -> [1, embed_dim, target_h, target_w]
    This directly computes the final expanded output from the input embedding
    """
    # 2D grid for output dimensions (target_h, target_w)  
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Create local workgroups for better memory efficiency
    local_h = tl.arange(0, BLOCK_SIZE)
    local_w = tl.arange(0, BLOCK_SIZE) 
    mask_h = local_h < target_h
    mask_w = local_w < target_w
    mask = mask_h[:, None] & mask_w[None, :]
    
    # For each output position and each embed dimension
    for embed_idx in range(embed_dim):
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Input coordinate: [seq_len, batch_size, embed_dim]
                in_offset = seq_idx * (batch_size * embed_dim) + batch_idx * embed_dim + embed_idx
                
                # Load input value
                input_val = tl.load(input_ptr + in_offset)
                
                # Broadcast input_val to entire target_h x target_w region
                # Output coordinate: [1, embed_dim, target_h, target_w]
                out_offset = embed_idx * (target_h * target_w) + pid_h * target_w + pid_w
                
                # Store the broadcasted value
                tl.store(output_ptr + out_offset, input_val, mask)

def triton_full_sequence_kernel(input_tensor, seq_len, batch_size, embed_dim, target_w, target_h):
    """
    Fusion: permute([2,0,1]) + unsqueeze(0) + expand((1,-1, target_h, target_w)) + contiguous
    """
    # Create output with final shape: [1, embed_dim, target_h, target_w]
    output_shape = (1, embed_dim, target_h, target_w)
    total_elements = embed_dim * target_h * target_w
    
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For efficiency, we'll use a simplified approach that handles the common case
    # where target_h is larger than seq_len (for position expansion)
    if target_h >= seq_len and target_w >= seq_len:
        # Efficient broadcasting approach
        BLOCK_SIZE = 16
        grid_h = (target_h + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_w = (target_w + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        triton_full_sequence_kernel_3d[(grid_h, grid_w)](
            input_ptr=input_tensor,
            output_ptr=output.view(-1),  # Flatten for easier indexing
            seq_len=seq_len,
            batch_size=batch_size, 
            embed_dim=embed_dim,
            target_w=target_w,
            target_h=target_h,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for different expansion patterns
        tmp_3 = input_tensor.permute([2, 0, 1])
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_5 = tmp_4.expand((1, -1, target_h, target_w))
        return tmp_5.contiguous()
    
    return output

def pattern(tmp_2, target_shape):
    # Pattern: permute([2, 0, 1]) -> unsqueeze(0) -> expand((1, -1, target_shape[0], target_shape[1]))
    # Note: This pattern matches the sequence that would go into contiguous()
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0) 
    tmp_5 = tmp_4.expand((1, -1, target_shape[0], target_shape[1]))
    return tmp_5

def replacement_args(tmp_2, target_shape):
    return (tmp_2, target_shape)

def replacement_func():
    return full_tensor_sequence_fusion