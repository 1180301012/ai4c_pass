import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match ReLU followed by flatten from dimension 1 to -1"""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(in_0):
    """Extract input tensor for the fused operation"""
    return (in_0,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Flatten kernel"""
    # Calculate total number of elements to process
    total_elements = batch_size * channels * height * width
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For [N, C, H, W] -> flatten(1, -1) -> [N, C*H*W]
    elements_per_batch = channels * height * width
    batch_idx = offsets // elements_per_batch
    remaining_in_batch = offsets % elements_per_batch
    
    # Calculate input offset (assuming tensor is contiguous)
    input_offset = batch_idx * elements_per_batch + remaining_in_batch
    
    # Load input data
    x = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store result (flattened layout)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(in_0):
    """Wrapper function for fused ReLU + Flatten operation"""
    input_shape = in_0.shape
    batch_size, channels, height, width = input_shape
    
    # Calculate output shape after flatten(1, -1)
    output_shape = (batch_size, channels * height * width)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # For fused operations, we can optimize block size based on operation complexity
    # Since we're doing both ReLU and flatten, this is worth the kernel overhead
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    total_elements = batch_size * channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure we have at least 1 program
    if num_programs == 0:
        num_programs = 1
    
    # Launch the fused kernel
    fused_relu_flatten_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused operation function"""
    return fused_relu_flatten