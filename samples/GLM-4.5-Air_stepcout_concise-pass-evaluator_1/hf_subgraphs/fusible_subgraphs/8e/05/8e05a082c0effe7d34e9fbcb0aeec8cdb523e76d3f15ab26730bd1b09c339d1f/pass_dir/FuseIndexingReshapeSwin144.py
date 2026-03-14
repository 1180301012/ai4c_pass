import torch
import triton
import triton.language as tl

# Pattern matching function for 144x144 pattern
def pattern(in_0, in_3):
    tmp_1 = in_0[in_3]
    tmp_2 = tmp_1.view(144, 144, -1)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_3):
    # Calculate the correct output shape by performing the actual operations first
    tmp_1 = in_0[in_3]
    tmp_2 = tmp_1.view(144, 144, -1)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    
    # Get the actual output shape and dimensions
    output_shape = tmp_5.shape
    height = 144
    width = 144
    n_elements = in_0.shape[1]
    
    return (in_0, in_3, height, width, n_elements, output_shape)

# Optimized kernel using Triton
@triton.jit
def fused_indexing_reshape_kernel(
    in_0_ptr,
    in_3_ptr, 
    out_ptr,
    table_height: tl.constexpr,
    table_width: tl.constexpr,
    grid_height: tl.constexpr,
    grid_width: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the output grid
    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    # Calculate output position
    output_offset = batch_pid * table_height * table_width * n_elements + pid * BLOCK_SIZE
    output_offsets = output_offset + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < table_height * table_width * n_elements
    
    # Load the index from in_3
    index = tl.load(in_3_ptr + pid, mask=None)
    
    # Calculate input position in the relative position bias table
    input_offset = index * n_elements
    input_offsets = input_offset + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < table_height * table_width * n_elements
    
    # Load the corresponding bias value from in_0
    bias_value = tl.load(in_0_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Store the result (this puts the bias value in the right position for the attention computation)
    tl.store(out_ptr + output_offsets, bias_value, mask=output_mask)

@torch.fx.wrap
def fused_indexing_reshape_swin(in_0, in_3, table_height, table_width, n_elements, output_shape):
    """
    Fused operation that combines indexing, view, permute, contiguous, and unsqueeze
    for Swin Transformer relative position bias computation.
    """
    # Get tensor sizes
    grid_height = in_3.shape[0]
    
    # Use the pre-computed output shape instead of calculating it
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Determine block size and launch grid
    BLOCK_SIZE = 1024
    grid_width = 1  # Only need one dimension since we're processing linearly
    num_programs = (grid_height * grid_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_indexing_reshape_kernel[(num_programs, 1)](
        in_0_ptr=in_0,
        in_3_ptr=in_3,
        out_ptr=out,
        table_height=table_height,
        table_width=table_width,
        grid_height=grid_height,
        grid_width=grid_width,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    def kernel_wrapper(in_0, in_3, table_height, table_width, n_elements, output_shape):
        return fused_indexing_reshape_swin(in_0, in_3, table_height, table_width, n_elements, output_shape)
    
    return kernel_wrapper