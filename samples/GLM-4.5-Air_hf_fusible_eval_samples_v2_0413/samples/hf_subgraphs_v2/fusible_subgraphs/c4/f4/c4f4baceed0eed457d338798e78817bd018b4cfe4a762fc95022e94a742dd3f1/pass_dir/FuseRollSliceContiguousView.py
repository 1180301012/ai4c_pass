import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    return tmp_3

def replacement_args(in_3):
    return (in_3, )

@triton.jit
def contiguous_view_kernel(
    input_ptr,
    output_ptr,
    original_shape_ptr,
    spatial_dim_0: tl.constexpr,
    spatial_dim_1: tl.constexpr,
    channel_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    program_idx = tl.program_id(0)
    offset = program_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total_elements = spatial_dim_0 * spatial_dim_1 * channel_dim
    mask = offset < total_elements
    
    # Calculate coordinates in the output tensor
    h_coords = (offset // (spatial_dim_1 * channel_dim)) % spatial_dim_0
    w_coords = (offset // channel_dim) % spatial_dim_1
    c_coords = offset % channel_dim
    
    # Calculate input index (assuming input was originally in the same layout)
    input_idx = offset
    
    # Load from input and store to output
    data = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offset, data, mask=mask)

@torch.fx.wrap
def optimized_contiguous_view(in_3):
    # Get input tensor shape
    original_batch, d1, d2, d3, d4, channels = in_3.shape
    
    # Determine spatial dimensions based on input tensor shape
    spatial_dim_0 = d1 * d2
    spatial_dim_1 = d3 * d4
    total_elements = spatial_dim_0 * spatial_dim_1 * channels
    
    # Calculate actual batch dimension
    batch_dim = original_batch  # Use the original batch dimension
    output_shape = (batch_dim, spatial_dim_0, spatial_dim_1, channels)
    output = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    contiguous_view_kernel[(num_programs,)](
        input_ptr=in_3,
        output_ptr=output,
        original_shape_ptr=in_3.shape,
        spatial_dim_0=spatial_dim_0,
        spatial_dim_1=spatial_dim_1,
        channel_dim=channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_contiguous_view