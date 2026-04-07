import torch
import triton
import triton.language as tl

@triton.jit
def position_encoding_kernel(
    input_ptr,  # [1, 15, 15, 512]
    weight_ptr,  # [12, 512]
    indices_ptr,  # [64, 64] -> flattened to [4096]
    output_ptr,  # [64, 64, 12]
    n_batch, n_height, n_width, n_features, n_embed,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread positions
    pid = tl.program_id(0)
    n_elements = n_height * n_width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int32)
    
    # Process each embedded dimension in parallel
    for embed_idx in range(n_embed):
        # Calculate input tensor offset for current embedded dimension
        input_offset = embed_idx * (n_batch * n_height * n_width * n_features)
        
        # Process each feature element in the block
        for feat_pid in range(tl.cdiv(n_features, BLOCK_SIZE)):
            feat_block_start = feat_pid * BLOCK_SIZE
            feat_offsets = feat_block_start + tl.arange(0, BLOCK_SIZE)
            feat_mask = feat_offsets < n_features
            
            # Load weight for current embedded dimension and feature offset
            weight_offset = embed_idx * n_features + feat_offsets
            weight = tl.load(weight_ptr + weight_offset, mask=feat_mask, other=0.0)
            
            # Load input features for all spatial positions in current block
            input_offsets = (indices[:, None] * n_features + feat_offsets[None, :]).flatten()
            input_mask = (offsets[:, None] < n_elements) & (feat_offsets[None, :] < n_features)
            input_data = tl.load(input_ptr + input_offsets.flatten(), mask=input_mask.flatten(), other=0.0)
            
            # Compute dot product
            sum_val = tl.sum(input_data * weight)
            
            # Store result at output position
            output_offset = (offsets * n_embed + embed_idx * BLOCK_SIZE + feat_offsets).flatten()
            output_mask = (offsets[:, None] < n_elements) & (feat_offsets[None, :] < n_features)
            tl.store(output_ptr + output_offset, sum_val, mask=output_mask.flatten())

@torch.fx.wrap
def optimized_position_encoding(input_tensor, weight_tensor, indices_tensor):
    # Input shapes
    n_batch, n_height, n_width, n_features = input_tensor.shape
    n_embed, _ = weight_tensor.shape
    
    # Output shape: [n_height, n_width, n_embed]
    output_shape = (n_height, n_width, n_embed)
    output = torch.zeros(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten indices
    indices_flat = indices_tensor.view(-1)  # [n_height * n_width]
    
    # Block sizes for optimal GPU utilization
    BLOCK_SIZE = 1024
    grid_size = (n_height * n_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    position_encoding_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        indices_ptr=indices_flat,
        output_ptr=output,
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        n_features=n_features,
        n_embed=n_embed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(linear_output, indices_flat):
    # Original pattern: linear -> view -> indexing
    tmp_3 = linear_output.view(-1, 12)  # [225, 12]
    tmp_5 = tmp_3[indices_flat]  # [4096, 12]
    result = tmp_5.view(64, 64, -1)  # [64, 64, 12]
    return result

def replacement_args(linear_output, indices_flat):
    return (linear_output, indices_flat)

def replacement_func():
    return optimized_position_encoding