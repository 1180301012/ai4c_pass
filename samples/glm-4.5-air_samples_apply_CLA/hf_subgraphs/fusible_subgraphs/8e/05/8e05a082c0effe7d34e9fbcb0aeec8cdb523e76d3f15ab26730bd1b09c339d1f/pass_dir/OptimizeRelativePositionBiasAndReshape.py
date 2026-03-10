import torch
import triton
import triton.language as tl

def pattern(in_0, in_3):
    tmp_1 = in_0[in_3]
    # Calculate the expected output shape based on input broadcasting needs
    # The result needs to be broadcastable with in_1 which is typically [heads, features, height, width]
    # So we create [1, features, height, width]
    spatial_size = tmp_1.numel() // in_0.size(1)  # total elements / features
    height = int(spatial_size ** 0.5)  # assuming square spatial layout
    width = height
    features = in_0.size(1)
    
    tmp_2 = tmp_1.view(height, width, features)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    return tmp_5

def replacement_args(in_0, in_3):
    return (in_0, in_3)

@triton.jit
def relative_position_bias_kernel(
    bias_table_ptr,
    indices_ptr,
    out_ptr,
    bias_height,
    bias_width,
    bias_features,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Calculate bias table coordinates
    bias_idx = indices
    out_idx = offsets
    
    # Load bias from table and store result directly
    bias_values = tl.load(bias_table_ptr + bias_idx * bias_features, mask=mask, other=0.0)
    
    # Reshape and permute in memory: from flat to [features, height, width]
    # We need to reinterpret the bias_values as [features, height, width]
    # where height x width = bias_height x bias_width = 144 x 144
    # and features = bias_features = 4
    
    # Store with proper dimension layout
    tl.store(out_ptr + out_idx * bias_features, bias_values, mask=mask)

@triton.jit
def relative_position_bias_kernel_2d(
    bias_table_ptr,
    indices_ptr,
    out_ptr,
    bias_height,
    bias_width,
    bias_features,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Calculate bias table coordinates
    bias_idx = indices
    
    # Load bias from table
    bias_values = tl.load(bias_table_ptr + bias_idx * bias_features, mask=mask, other=0.0)
    
    # Store result with proper layout: [total_elements, features]
    tl.store(out_ptr + offsets * bias_features, bias_values, mask=mask)

@torch.fx.wrap
def optimized_relative_position_bias(bias_table, indices):
    # Calculate dimensions dynamically
    features = bias_table.size(1)
    spatial_size = indices.size(0)  # Total number of spatial positions
    height = int(spatial_size ** 0.5)  # Assuming square spatial layout
    width = height
    
    total_elements = height * width
    num_blocks = (total_elements + 1023) // 1024
    BLOCK_SIZE = 1024
    
    # Create 2D intermediate tensor: [height*width, features]
    intermediate = torch.empty((total_elements, features), dtype=torch.float32, device=bias_table.device)
    
    # Launch kernel to fill intermediate tensor
    relative_position_bias_kernel_2d[(num_blocks,)](
        bias_table,
        indices,
        intermediate,
        height,
        width,
        features,
        total_elements,
        BLOCK_SIZE,
    )
    
    # Reshape to [features, height, width] then add batch dimension
    result = intermediate.view(features, height, width).unsqueeze(0)
    
    return result

def replacement_func():
    return optimized_relative_position_bias