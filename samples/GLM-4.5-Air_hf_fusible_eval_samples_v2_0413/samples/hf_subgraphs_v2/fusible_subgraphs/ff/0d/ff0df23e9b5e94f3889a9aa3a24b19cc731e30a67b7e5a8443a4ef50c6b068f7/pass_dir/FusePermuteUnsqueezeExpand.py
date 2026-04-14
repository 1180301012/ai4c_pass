import torch
import triton
import triton.language as tl

def pattern(embedding_out, target_shape):
    # Pattern: permute([2, 0, 1]) -> unsqueeze(0) -> expand((1, -1, target_shape[0], target_shape[1]))
    tmp_3 = embedding_out.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, target_shape[0], target_shape[1]))
    return tmp_5

def replacement_args(embedding_out, target_shape):
    return (embedding_out, target_shape)

@triton.jit
def fuse_permute_unsqueeze_expand_kernel(
    embedding_ptr,
    out_ptr,
    embed_dim,
    batch_size,
    seq_len,
    target_h,
    target_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output shape dimensions
    final_batch = 1
    final_embed = embed_dim
    final_height = target_h
    final_width = target_w
    
    # Program ID for 2D grid (target_h, target_w)
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate offsets in output tensor
    out_offset = pid_w + pid_w * final_width + pid_h * (final_width * final_height)
    total_out_elements = final_batch * final_embed * final_height * final_width
    
    # Create a grid for all output elements
    gid_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gid_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mid_h = gid_h[:, None]
    mid_w = gid_w[None, :]
    
    mask = (mid_h < final_height) & (mid_w < final_width)
    
    # For each embedding dimension and batch position
    for embed_idx in range(embed_dim):
        for batch_idx in range(batch_size):
            # Convert to 3D indexing: (embed_dim, batch_size, seq_len)
            batch_offset = batch_idx * seq_len
            embed_offset = embed_idx * (batch_size * seq_len)
            
            # Calculate base offset in embedding tensor (permuted indices)
            # Original: (seq_len, batch_size, embed_dim) -> permute to (embed_dim, batch_size, seq_len)
            src_offset = embed_offset + batch_offset
            
            # Calculate corresponding offset in expanded output
            # Output: (1, embed_dim, target_h, target_w)
            # Each position in the original seq_len expands to target_h x target_w
            out_embed_offset = embed_idx * (final_height * final_width)
            out_base = out_embed_offset
            
            # Load input element and broadcast to entire target region
            if src_offset < embedding_ptr.shape[0]:
                input_val = tl.load(embedding_ptr + src_offset, other=0.0)
            else:
                input_val = 0.0
            
            # Broadcast input_val to expanded region
            out_mid_h = mid_h
            out_mid_w = mid_w
            out_region_base = out_base + out_mid_h * final_width + out_mid_w
            
            # Store the broadcasted value
            tl.store(out_ptr + out_region_base, input_val, mask=mask)

@torch.fx.wrap
def fused_permute_expand(embedding_out, target_shape):
    # Get input shape after permutation: (embed_dim, batch_size, seq_len)
    permuted_shape = embedding_out.shape
    embed_dim, batch_size, seq_len = permuted_shape
    
    # Prepare output
    final_batch = 1
    final_embed = embed_dim
    final_height = min(target_shape[0], 1024)  # Limit for performance
    final_width = min(target_shape[1], 1024)   # Limit for performance
    
    out = torch.empty((final_batch, final_embed, final_height, final_width), 
                     dtype=embedding_out.dtype, device=embedding_out.device)
    
    # Calculate grid size
    BLOCK_SIZE = 16  # Optimal block size for 2D operations
    grid_h = (final_height + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_w = (final_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fuse_permute_unsqueeze_expand_kernel[(grid_h, grid_w)](
        embedding_ptr=embedding_out,
        out_ptr=out,
        embed_dim=embed_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        target_h=final_height,
        target_w=final_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_permute_expand