import torch
import triton
import triton.language as tl

# Pattern matching function - matches tile+cat+add operations
def pattern(conv_out_flat_transposed, cls_token, position_embeddings):
    # cls_token: [1, 1, 768] needs to be tiled to match spatial size
    tmp_9 = cls_token.tile([1, 1, 1])  # tile to [1, 233435, 768] to match spatial dims
    tmp_10 = torch.cat((tmp_9, conv_out_flat_transposed), dim=1)  # concat along dim=1
    result = tmp_10 + position_embeddings  # add position embeddings
    return tmp_10, result

# Argument extraction function
def replacement_args(conv_out_flat_transposed, cls_token, position_embeddings):
    return (conv_out_flat_transposed, cls_token, position_embeddings)

# Optimized Triton kernel for embedding concatenation
@triton.jit
def embedding_concat_kernel(
    conv_out_flat_ptr, cls_token_ptr, pos_emb_ptr,
    intermediate_out_ptr, final_out_ptr,
    batch_size, channels, spatial_size, pos_emb_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel processing
    pid = tl.program_id(0)
    total_elements = batch_size * spatial_size
    
    # Each block handles a portion of the spatial dimension
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load flattened conv3d output (already transposed to [spatial_size, channels])
    conv_out = tl.load(conv_out_flat_ptr + offsets * channels, mask=mask, other=0.0)
    
    # Load cls_token and broadcast to spatial dimensions
    cls_token = tl.load(cls_token_ptr + tl.arange(0, channels), mask=tl.arange(0, channels) < channels, other=0.0)
    cls_expanded = cls_token + tl.zeros((BLOCK_SIZE, channels), dtype=tl.float16)
    
    # Store intermediate result: concatenate cls_token with conv_out
    # Each spatial location gets cls_token followed by conv_out
    tl.store(intermediate_out_ptr + (offsets * 2) * channels, cls_expanded, mask=mask)
    tl.store(intermediate_out_ptr + (offsets * 2 + 1) * channels, conv_out, mask=mask)
    
    # Load position embeddings
    pos_emb = tl.load(pos_emb_ptr + tl.arange(0, channels), mask=tl.arange(0, channels) < channels, other=0.0)
    pos_expanded = pos_emb + tl.zeros((BLOCK_SIZE, channels), dtype=tl.float16)
    
    # Add position embeddings to intermediate result (just use cls_token part)
    cls_part = tl.load(intermediate_out_ptr + (offsets * 2) * channels, mask=mask, other=0.0)
    result = cls_part + pos_expanded
    
    tl.store(final_out_ptr + offsets * channels, result, mask=mask)

@torch.fx.wrap
def optimized_embedding_concat(conv_out_flat_transposed, cls_token, position_embeddings):
    # Get tensor shapes
    batch_size = conv_out_flat_transposed.shape[0]
    spatial_size = conv_out_flat_transposed.shape[0]  # [1, 233435, 768] -> spatial_size = 233435
    channels = 768
    
    # Calculate output shapes
    intermediate_length = spatial_size * 2  # [1, 466870, 768]
    
    # Output tensors
    intermediate_out = torch.empty((1, intermediate_length, channels), dtype=conv_out_flat_transposed.dtype, device=conv_out_flat_transposed.device)
    final_out = torch.empty((1, channels), dtype=conv_out_flat_transposed.dtype, device=conv_out_flat_transposed.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    embedding_concat_kernel[(num_programs,)](
        conv_out_flat_ptr=conv_out_flat_transposed,
        cls_token_ptr=cls_token,
        pos_emb_ptr=position_embeddings,
        intermediate_out_ptr=intermediate_out,
        final_out_ptr=final_out,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        pos_emb_len=position_embeddings.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return intermediate_out, final_out

# Replacement function
def replacement_func():
    return optimized_embedding_concat