import torch
import triton
import triton.language as tl

def mean_with_final_pattern(input_tensor):
    """
    Pattern matching for mean reduction followed by view operation.
    This is: tmp_1 = tmp_0.mean((2, 3)) followed by tmp_4 = tmp_1.view(1, 1, -1)
    
    The mean reduction produces [1, C, 1, 1] and view reshapes to [1, 1, C].
    We can optimize this by computing the mean directly into the final shape.
    """
    tmp_1 = input_tensor.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_4

def replacement_args(input_tensor):
    # Extract the input tensor
    return (input_tensor,)

@triton.jit
def optimized_mean_view_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that computes mean reduction directly into final view shape.
    This avoids the intermediate [1, C, 1, 1] tensor and directly produces [1, 1, C].
    """
    # Program identifiers
    pid_m = tl.program_id(0)
    
    # Compute offsets for channels
    m_offset = pid_m * BLOCK_SIZE_M
    
    # Calculate bounds
    m_mask = m_offset < n_channels
    
    if m_mask:
        # Load input data for the entire channel
        input_ptrs = input_ptr + m_offset * height * width
        input_data = tl.load(input_ptrs + tl.arange(0, height * width), other=0.0)
        
        # Compute mean reduction directly to final shape
        channel_sum = tl.sum(input_data)
        channel_mean = channel_sum / (height * width)
        
        # Store result directly in final view shape [1, 1, C]
        # output_ptr points to [0, 0, channel_index] 
        output_index = m_offset
        tl.store(output_ptr + output_index, channel_mean)

@torch.fx.wrap
def optimized_mean_view(input_tensor):
    """
    Wrapper function that computes mean reduction directly into final view shape.
    Input shape: [1, C, H, W]
    Output shape: [1, 1, C] (equivalent to .mean((2, 3)).view(1, 1, -1))
    """
    # Unpack tensor dimensions
    batch_size, channels, height, width = input_tensor.shape
    assert batch_size == 1, "Only batch size 1 is supported"
    
    # Create output tensor directly in final view shape [1, 1, C]
    output = torch.zeros(1, 1, channels, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 128  # Number of channels per block (larger blocks for better occupancy)
    
    # Calculate grid dimensions
    grid_m = (channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch the optimized kernel
    optimized_mean_view_kernel[(grid_m,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    """Returns the optimized mean + view function"""
    return optimized_mean_view