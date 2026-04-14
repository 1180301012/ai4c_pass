import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_elements_per_instance,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one instance (batch * channel) with BLOCK_SIZE elements
    pid = tl.program_id(0)
    instance_offset = pid * n_elements_per_instance
    
    # Process elements in blocks within this instance
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = instance_offset + block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (instance_offset + n_elements_per_instance)
    
    # Linear indexing within the height x width matrix
    linear_indices = offsets - instance_offset
    
    # Convert linear indices to (height, width) coordinates
    h_indices = linear_indices // width
    w_indices = linear_indices % width
    
    # Transpose coordinates
    transposed_h = w_indices
    transposed_w = h_indices
    
    # Calculate transposed linear indices
    transposed_linear = transposed_h * width + transposed_w
    transposed_offsets = instance_offset + transposed_linear
    
    # Load original values and store to transposed positions
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + transposed_offsets, x_values, mask=mask)

@torch.fx.wrap
def transpose_wrapper(x):
    # Get tensor dimensions for [batch, channels, height, width]
    batch, channels, height, width = x.shape
    
    # Calculate number of elements per instance (channel)
    n_elements_per_instance = height * width
    total_instances = batch * channels
    
    # Use optimal block size for better performance
    BLOCK_SIZE = 1024  # Good balance between occupancy and memory efficiency
    inner_blocks = triton.cdiv(n_elements_per_instance, BLOCK_SIZE)
    
    # Create grid: (total_instances, inner_blocks)
    grid = (total_instances, inner_blocks)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the optimized transpose kernel
    transpose_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements_per_instance=n_elements_per_instance,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return transpose_wrapper