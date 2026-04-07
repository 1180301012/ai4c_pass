import torch
import triton
import triton.language as tl

@triton.jit
def simple_linear_kernel(
    input_ptr,      # [batch, height, width, features] - input tensor
    weight_ptr,     # [embed_dim, features] - weight matrix  
    indices_ptr,    # [grid_height, grid_width] - position indices
    output_ptr,     # [grid_height, grid_width, embed_dim] - final output
    batch, height, width, features, embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = height * width
    
    if pid * BLOCK_SIZE >= total_elements:
        return
        
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Process each embedding dimension separately
    for embed_i in range(embed_dim):
        # Get position indices for this thread
        pos_idx = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int32)
        
        # Simple dot product computation for one element at a time
        for i in range(tl.sum(mask)):
            pos = pos_idx[i]
            if pos < batch * height * width:
                # Load weight for this embedding dimension
                weight_offset = embed_i * features + tl.arange(0, features)
                weight = tl.load(weight_ptr + weight_offset, mask=weight_offset < features, other=0.0)
                
                # Load input features for this position
                input_offset = pos * features + tl.arange(0, features)
                input_data = tl.load(input_ptr + input_offset, mask=input_offset < features, other=0.0)
                
                # Compute dot product
                result = tl.sum(input_data * weight)
                
                # Store result
                output_offset = offsets[i] * embed_dim + embed_i
                tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def optimized_linear_indexing(input_tensor, weight_tensor, indices_tensor):
    # Input shapes
    batch, height, width, features = input_tensor.shape
    embed_dim, _ = weight_tensor.shape
    
    # Output shape: [height, width, embed_dim]
    output_shape = (height, width, embed_dim)
    output = torch.zeros(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch simple kernel
    BLOCK_SIZE = 1024
    grid_size = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_linear_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        indices_ptr=indices_tensor,
        output_ptr=output,
        batch=batch,
        height=height,
        width=width,
        features=features,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(input_tensor, weight_tensor, indices_result):
    # The exact pattern from the original computation:
    # linear = torch.nn.functional.linear(input_tensor, weight_tensor, None)
    linear = torch.nn.functional.linear(input_tensor, weight_tensor, None)
    tmp_3 = linear.view(-1, 12)
    tmp_4 = indices_result.view(-1)
    tmp_5 = tmp_3[tmp_4]
    result = tmp_5.view(64, 64, -1)
    return result

def replacement_args(input_tensor, weight_tensor, indices_result):
    # Return all three arguments needed for the optimized function
    # Note: input_tensor is the input to linear, weight_tensor is the linear weight
    return (input_tensor, weight_tensor, indices_result)

def replacement_func():
    return optimized_linear_indexing