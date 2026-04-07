import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching for two-input addition chain + spatial mean
    tmp_0 = 0 + in_1; tmp_0 += in_0; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    """
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement kernel"""
    return (in_0, in_1)

@triton.jit
def optimized_two_input_add_mean_kernel(
    in0_ptr, in1_ptr,
    sum_out_ptr, mean_out_ptr,
    batch_size, num_channels, height, width,
    dtype: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    """
    Optimized kernel for two-input addition chain + spatial mean computation
    """
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Calculate offsets
    batch_offset = pid_n * num_channels * height * width
    channel_offset = pid_c * height * width
    spatial_offset = pid_hw * BLOCK_SIZE_HW
    
    global_offset = batch_offset + channel_offset + spatial_offset
    
    # Mask for spatial bounds
    spatial_mask = spatial_offset + tl.arange(0, BLOCK_SIZE_HW) < height * width
    
    # Load two input tensors
    in0_data = tl.load(in0_ptr + global_offset, mask=spatial_mask, other=0.0)
    in1_data = tl.load(in1_ptr + global_offset, mask=spatial_mask, other=0.0)
    
    # Perform fused addition: in0_data + in1_data
    sum_result = in0_data + in1_data
    
    # Store sum result
    tl.store(sum_out_ptr + global_offset, sum_result, mask=spatial_mask)

@torch.fx.wrap
def optimized_two_input_add_mean(in_0, in_1):
    """
    Optimized function for two-input addition chain with spatial mean
    """
    # Get input tensor properties
    first_input = in_0
    shape = first_input.shape
    dtype = first_input.dtype
    device = first_input.device
    
    if len(shape) != 4:
        raise ValueError(f"Expected 4D tensor, got {len(shape)}D")
    
    batch_size, num_channels, height, width = shape
    
    # Create output tensors
    sum_out = torch.empty_like(first_input)
    mean_out = torch.empty((batch_size, num_channels, 1, 1), dtype=dtype, device=device)
    
    # Set up kernel launch configuration
    BLOCK_SIZE_HW = 1024  # Block size for spatial dimensions
    
    # Calculate grid dimensions
    spatial_elements = height * width
    num_hw_blocks = (spatial_elements + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    num_channel_blocks = (num_channels + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    num_batch_blocks = (batch_size + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch Triton kernel for sum computation
    optimized_two_input_add_mean_kernel[
        (num_batch_blocks, num_channel_blocks, num_hw_blocks)
    ](
        in_0, in_1,
        sum_out, mean_out,
        batch_size, num_channels, height, width,
        dtype,
        BLOCK_SIZE_HW
    )
    
    # Compute mean using PyTorch (can be optimized with Triton later)
    mean_result = sum_out.mean((2, 3), keepdim=True)
    
    return sum_out, mean_result

def replacement_func():
    """Return the optimized replacement function"""
    return optimized_two_input_add_mean