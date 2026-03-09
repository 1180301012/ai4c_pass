import torch
import triton
import triton.language as tl

def pattern(*args):
    # The framework passes additional context arguments, we only need the first two
    bias_table = args[0]
    indices = args[1]
    tmp_1 = bias_table[indices]
    tmp_2 = tmp_1.view(-1, -1, tmp_1.shape[-1])  # Flexible spatial dimensions
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    return tmp_5

def replacement_args(*args):
    # We only need the bias_table and indices
    return (args[0], args[1])

@triton.jit
def fused_position_bias_kernel(
    bias_table_ptr,
    indices_ptr,
    output_ptr,
    num_indices,
    bias_height,
    bias_width,
    num_heads,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = bias_height * bias_width * num_heads
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Calculate indices for each dimension
    heads = offset // (bias_height * bias_width)
    remaining = offset % (bias_height * bias_width)
    y_coords = remaining // bias_width
    x_coords = remaining % bias_width
    
    # Load bias table indices
    bias_indices = tl.load(bias_table_ptr + indices_ptr * tl.max(bias_height, bias_width) + y_coords * bias_width + x_coords, mask=mask, other=0)
    bias_values = tl.load(bias_table_ptr + bias_indices * num_heads + heads, mask=mask, other=0.0)
    
    # Store output (already in correct shape due to indexing order)
    tl.store(output_ptr + offset, bias_values, mask=mask)

@torch.fx.wrap
def fused_position_bias_forward(bias_table, indices):
    """
    Optimized fusion of indexing + view + permute + contiguous + unsqueeze
    """
    # Get output shape from the expected computation
    # The output should be (1, num_heads, height, width)
    # We can determine the spatial size from indices shape
    
    # For most transformers, the spatial size is sqrt(indices.shape[0])
    # Let's compute spatial dimensions dynamically
    total_spatial_elements = indices.shape[0]
    spatial_size = int(total_spatial_elements ** 0.5)
    height = width = spatial_size
    
    # The number of heads is determined by the bias_table shape
    num_heads = bias_table.shape[1]
    
    total_elements = height * width * num_heads
    output = torch.empty((1, num_heads, height, width), dtype=bias_table.dtype, device=bias_table.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_position_bias_kernel[(num_programs,)](
        bias_table_ptr=bias_table,
        indices_ptr=indices,
        output_ptr=output,
        num_indices=indices.numel(),
        bias_height=height,
        bias_width=width,
        num_heads=num_heads,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_position_bias_forward