import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches the fused computation: in_1 + in_2 + in_0 followed by mean computation
    """
    # tmp_0 = in_1 + in_2
    tmp_0 = in_1 + in_2
    # tmp_0 += in_0
    tmp_0 += in_0
    # Identity operation and mean computation
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for kernel replacement
    """
    inputs = [in_0, in_1, in_2]
    return tuple(inputs)

@triton.jit
def fused_add_mean_kernel(
    input_ptrs,  # List of input tensor pointers
    out_sum_ptr,  # Output sum tensor pointer  
    out_mean_ptr, # Output mean tensor pointer
    n_channels,   # Number of channels
    height,       # Height dimension
    width,        # Width dimension
    num_inputs: tl.constexpr,  # Number of input tensors (should be 3)
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel processing
):
    """
    Optimized Triton kernel that fuses addition and mean computation
    """
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate total spatial elements per channel
    spatial_elements = height * width
    total_elements = n_channels * spatial_elements
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, total_elements)
    
    # Process each spatial location
    for idx in range(block_start, block_end):
        # Convert linear index to 3D indices (channel, y, x)
        chan_idx = idx // spatial_elements
        spatial_idx = idx % spatial_elements
        y_idx = spatial_idx // width
        x_idx = spatial_idx % width
        
        # Initialize accumulator
        sum_val = 0.0
        
        # Sum all input tensors at this location
        for i in range(num_inputs):
            # Calculate offset for current channel and spatial location
            offset = (chan_idx * spatial_elements + spatial_idx)
            val = tl.load(input_ptrs[i] + offset, other=0.0)
            sum_val += val
        
        # Store summed result
        out_offset = chan_idx * spatial_elements + spatial_idx
        tl.store(out_sum_ptr + out_offset, sum_val)
        
        # Compute mean for the spatial dimensions (height, width)
        mean_val = sum_val / (height * width)
        mean_offset = chan_idx  # Output mean has shape [N, C, 1, 1]
        tl.store(out_mean_ptr + mean_offset, mean_val)

@torch.fx.wrap
def fused_add_mean_kernel_wrapper(input_tensors):
    """
    Wrapper function to launch the fused kernel
    """
    # Determine number of inputs
    num_inputs = len(input_tensors)
    
    # Get input tensor shapes (all inputs have same shape)
    input_shape = input_tensors[0].shape
    batch_size, n_channels, height, width = input_shape
    
    # Create output tensors
    out_sum = torch.empty_like(input_tensors[0])
    out_mean = torch.empty((batch_size, n_channels, 1, 1), device=input_tensors[0].device, dtype=input_tensors[0].dtype)
    
    # Flatten input tensor pointers
    input_ptrs = [t.contiguous().data_ptr() for t in input_tensors]
    sum_ptr = out_sum.data_ptr()
    mean_ptr = out_mean.data_ptr()
    
    # Configure kernel launch
    total_elements = batch_size * n_channels * height * width
    BLOCK_SIZE = 1024  # Optimized block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_mean_kernel[(num_programs,)](
        input_ptrs,
        sum_ptr,
        mean_ptr,
        batch_size * n_channels,  # Total channels across batch
        height,
        width,
        num_inputs,
        BLOCK_SIZE,
    )
    
    return out_sum, out_mean

def replacement_func():
    """
    Returns the optimized kernel function
    """
    return fused_add_mean_kernel_wrapper