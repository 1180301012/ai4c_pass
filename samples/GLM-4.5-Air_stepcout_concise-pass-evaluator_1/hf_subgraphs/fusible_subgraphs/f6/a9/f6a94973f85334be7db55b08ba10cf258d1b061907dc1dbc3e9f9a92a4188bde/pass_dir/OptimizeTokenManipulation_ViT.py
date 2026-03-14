import torch
import triton
import triton.language as tl

def pattern(arg0, arg1, arg2, arg3, arg4, class_token, patch_tokens, pos_embed):
    # Pattern matching: expand + concat + add sequence
    # This represents the class token manipulation in ViT
    expanded_class_token = class_token.expand(1, -1, -1)
    concatenated = torch.cat([expanded_class_token, patch_tokens], dim=1)
    result = concatenated + pos_embed
    return result

def replacement_args(arg0, arg1, arg2, arg3, arg4, class_token, patch_tokens, pos_embed):
    return (class_token, patch_tokens, pos_embed)

@triton.jit
def token_manipulation_kernel(
    class_token_ptr,      # [1, 1, embed_dim]
    patch_tokens_ptr,     # [1, num_patches, embed_dim]
    pos_embed_ptr,        # [1, num_patches + 1, embed_dim]
    output_ptr,           # [1, num_patches + 1, embed_dim]
    num_patches,          # Number of patch tokens
    embed_dim,            # Embedding dimension
    BLOCK_SIZE_M: tl.constexpr,      # Block size for sequence dimension
    BLOCK_SIZE_N: tl.constexpr,      # Block size for embedding dimension
):
    # Each program handles a block of tokens
    pid_m = tl.program_id(0)  # Sequence position (0 = class token, 1-num_patches = patch tokens)
    pid_n = tl.program_id(1)  # Embedding dimension
    
    # Total sequence length
    total_seq_len = num_patches + 1
    
    # Initialize output pointer for this program
    output_base = output_ptr + pid_m * embed_dim + pid_n
    
    # Process in blocks for better memory efficiency
    for n_base in tl.range(0, embed_dim, BLOCK_SIZE_N):
        n_offsets = n_base + tl.arange(0, BLOCK_SIZE_N)
        if n_offsets < embed_dim:
            if pid_m == 0:
                # For class token position (first position)
                class_val = tl.load(class_token_ptr + n_offsets, mask=(n_offsets < embed_dim), other=0.0)
                pos_val = tl.load(pos_embed_ptr + n_offsets, mask=(n_offsets < embed_dim), other=0.0)
                tl.store(output_base + n_offsets, class_val + pos_val, mask=(n_offsets < embed_dim))
            else:
                # For patch token positions
                patch_val = tl.load(patch_tokens_ptr + (pid_m - 1) * embed_dim + n_offsets, mask=(n_offsets < embed_dim), other=0.0)
                pos_val = tl.load(pos_embed_ptr + pid_m * embed_dim + n_offsets, mask=(n_offsets < embed_dim), other=0.0)
                tl.store(output_base + n_offsets, patch_val + pos_val, mask=(n_offsets < embed_dim))

@torch.fx.wrap
def optimized_token_manipulation(class_token, patch_tokens, pos_embed):
    # Get tensor dimensions
    batch_size, class_seq_len, embed_dim = class_token.shape
    patch_batch, patch_seq_len, patch_embed_dim = patch_tokens.shape
    pos_batch, pos_seq_len, pos_embed_dim = pos_embed.shape
    
    # Verify dimensions match
    assert batch_size == patch_batch == pos_batch, f"Batch size mismatch: {batch_size} vs {patch_batch} vs {pos_batch}"
    assert embed_dim == patch_embed_dim == pos_embed_dim, f"Embed dimension mismatch: {embed_dim} vs {patch_embed_dim} vs {pos_embed_dim}"
    assert class_seq_len == 1, f"Class token should have sequence length 1, got {class_seq_len}"
    assert pos_seq_len == patch_seq_len + 1, f"Pos embed length mismatch: {pos_seq_len} vs {patch_seq_len + 1}"
    
    num_patches = patch_seq_len
    total_seq_len = num_patches + 1
    
    # Create output tensor
    output = torch.empty((batch_size, total_seq_len, embed_dim), dtype=class_token.dtype, device=class_token.device)
    
    # Choose optimal block sizes
    BLOCK_SIZE_M = min(128, total_seq_len)  # Sequence dimension block size
    BLOCK_SIZE_N = min(32, embed_dim)       # Embedding dimension block size
    
    # Calculate grid dimensions
    grid = (total_seq_len, (embed_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch Triton kernel
    token_manipulation_kernel[grid](
        class_token,
        patch_tokens,
        pos_embed,
        output,
        num_patches,
        embed_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_token_manipulation