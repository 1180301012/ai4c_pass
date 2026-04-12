import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    """Match flatten from dimension 1 to -1"""
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(tmp_0):
    """Extract input tensor for the flatten operation"""
    return (tmp_0,)

@triton.jit
def optimized_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simplified and correct flatten kernel"""
    # Calculate total number of elements to process
    # For flatten(1, -1): [N, C, H, W] -> [N, C*H*W]
    total_elements = batch_size * channels * height * width
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For flatten(1, -1) on [N, C, H, W] tensors, the memory layout is:
    # The original tensor is already laid out in memory as [N, C, H, W]
    # We need to flatten from dimension 1, so we preserve the batch dimension
    # and flatten channels, height, and width into one dimension
    
    # Calculate batch index and remaining offset
    # total_elements_per_batch = channels * height * width
    elements_per_batch = channels * height * width
    batch_idx = offsets // elements_per_batch
    remaining_in_batch = offsets % elements_per_batch
    
    # Calculate the flat index within the flattened channels*height*width
    # For [N, C, H, W] -> [N, C*H*W], the source index is:
    # source_offset = batch_idx * channels * height * width + remaining_in_batch
    # But since our input tensor is contiguous in memory, we can simplify this
    source_offset = batch_idx * elements_per_batch + remaining_in_batch
    
    # Load from source tensor
    x = tl.load(x_ptr + source_offset, mask=mask, other=0.0)
    
    # Store to output tensor (flattened layout is already contiguous)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_flatten(tmp_0):
    """Wrapper function for the optimized flatten operation"""
    input_shape = tmp_0.shape
    batch_size, channels, height, width = input_shape
    
    # Calculate output shape after flatten(1, -1)
    output_shape = (batch_size, channels * height * width)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Calculate optimal block size based on input size
    BLOCK_SIZE = 1024  # Good balance for most GPU architectures
    
    # Calculate number of programs needed
    total_elements = batch_size * channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure we have at least 1 program
    if num_programs == 0:
        num_programs = 1
    
    # Launch the kernel
    optimized_flatten_kernel[(num_programs,)](
        x_ptr=tmp_0,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized operation function"""
    return optimized_flatten