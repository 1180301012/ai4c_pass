import torch
import triton
import triton.language as tl

# Pattern matching function - matches flatten+transpose+concat operations
def pattern(conv_out, cls_token, position_embeddings):
    # conv_out: [1, 768, 5, 223, 209] from conv3d
    tmp_7 = conv_out.flatten(2)  # flatten dims 2,3,4 -> [1, 768, 233435]
    tmp_8 = tmp_7.transpose(1, 2)  # [1, 233435, 768]
    tmp_9 = cls_token.tile([1, 1, 1])  # [1, 1, 768] tiled to [1, 233435, 768]
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)  # [1, 466870, 768]
    result = tmp_10 + position_embeddings  # [1, 981, 768] broadcasted addition
    return tmp_10, result

# Argument extraction function
def replacement_args(conv_out, cls_token, position_embeddings):
    return (conv_out, cls_token, position_embeddings)

# Optimized Triton kernel for fused operations
@triton.jit
def fused_kernel(
    conv_out_ptr, cls_token_ptr, position_embeddings_ptr,
    out_ptr, intermediate_ptr,
    batch_size, channels, spatial_size, cls_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel processing
    pid = tl.program_id(0)
    
    # Compute total spatial elements: 5 * 223 * 209 = 233435
    total_elements = spatial_size
    
    # Each block handles a portion of the spatial dimension
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load conv3d output and flatten to [total_elements, channels]
    conv_out = tl.load(conv_out_ptr + offsets * channels, mask=mask, other=0.0)
    
    # Load and tile cls_token to match spatial size
    cls_token_val = tl.load(cls_token_ptr + tl.arange(0, channels), mask=tl.arange(0, channels) < channels, other=0.0)
    cls_token_expanded = cls_token_val + tl.zeros((BLOCK_SIZE, channels), dtype=tl.float16)
    
    # Concatenate cls_token with flattened conv_out
    conv_out_expanded = conv_out + tl.zeros((BLOCK_SIZE, channels), dtype=tl.float16)
    
    # Store concatenated result (cls_token + conv_out)
    tl.store(intermediate_ptr + (offsets * 2) * channels, cls_token_expanded, mask=mask)
    tl.store(intermediate_ptr + (offsets * 2 + 1) * channels, conv_out_expanded, mask=mask)
    
    # Load position embeddings and add
    pos_emb = tl.load(position_embeddings_ptr + tl.arange(0, channels), mask=tl.arange(0, channels) < channels, other=0.0)
    pos_emb_expanded = pos_emb + tl.zeros((BLOCK_SIZE, channels), dtype=tl.float16)
    
    # Add position embeddings to concatenated result
    intermediate_data = tl.load(intermediate_ptr + offsets * 2 * channels, mask=mask, other=0.0)
    final_result = intermediate_data + pos_emb_expanded
    
    tl.store(out_ptr + offsets * channels, final_result, mask=mask)

@torch.fx.wrap
def fused_operation(conv_out, cls_token, position_embeddings):
    # Get tensor shapes
    batch_size = conv_out.shape[0]
    channels = conv_out.shape[1]
    
    # Calculate spatial size: 5 * 223 * 209 = 233435
    spatial_size = conv_out.shape[2] * conv_out.shape[3] * conv_out.shape[4]
    cls_size = cls_token.shape[1]
    
    # Calculate output sizes
    intermediate_shape = (1, 2 * spatial_size, channels)  # [1, 466870, 768]
    
    # Output tensors
    intermediate_out = torch.empty((1, 2 * spatial_size, channels), dtype=conv_out.dtype, device=conv_out.device)
    final_out = torch.empty((batch_size, channels), dtype=conv_out.dtype, device=conv_out.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_kernel[(num_programs,)](
        conv_out_ptr=conv_out,
        cls_token_ptr=cls_token,
        position_embeddings_ptr=position_embeddings,
        out_ptr=final_out,
        intermediate_ptr=intermediate_out,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        cls_size=cls_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return intermediate_out, final_out

# Replacement function
def replacement_func():
    return fused_operation