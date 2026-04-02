import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern matching: SiLU -> Adaptive AvgPool2d(1,1) -> Flatten"""
    tmp_0 = torch.nn.functional.silu(input_tensor, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2

def replacement_args(input_tensor):
    """Extract arguments needed for replacement"""
    return (input_tensor,)

@triton.jit
def fused_silu_spatial_mean_kernel(
    input_ptr,
    output_ptr,
    batch_idx,
    n_channels,
    spatial_height,
    spatial_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that applies SiLU and computes spatial mean per channel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_channels
    
    # Calculate offset to current batch
    batch_offset = batch_idx * n_channels * spatial_height * spatial_width
    
    # Load all spatial elements for this channel
    elements_per_channel = spatial_height * spatial_width
    all_elements = tl.zeros(elements_per_channel, dtype=tl.float32)
    
    # Load all elements for this channel across spatial dimensions
    for i in range(elements_per_channel):
        element_offset = batch_offset + offsets * elements_per_channel + i
        val = tl.load(input_ptr + element_offset, mask=mask, other=0.0)
        all_elements = tl.where(offsets < n_channels, val, all_elements)
    
    # Compute spatial mean: sum all spatial elements and divide by count
    channel_sum = tl.sum(all_elements)
    spatial_mean = channel_sum / elements_per_channel
    
    # Apply SiLU activation to the mean result
    sigmoid_mean = 1.0 / (1.0 + tl.exp(-spatial_mean))
    silu_mean = spatial_mean * sigmoid_mean
    
    # Store result
    tl.store(output_ptr + offsets, silu_mean, mask=mask)

@torch.fx.wrap
def fused_silu_spatial_mean(input_tensor):
    """Compute fused SiLU spatial mean using Triton"""
    # Get input dimensions
    batch_size, n_channels, spatial_height, spatial_width = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((batch_size, n_channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size for kernel launch - adjust based on channels
    BLOCK_SIZE = 128
    num_programs = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each batch separately (simpler and more memory efficient)
    for b in range(batch_size):
        batch_offset = b * n_channels * spatial_height * spatial_width
        output_offset = b * n_channels
        
        fused_silu_spatial_mean_kernel[(num_programs,)](
            input_ptr=input_tensor + batch_offset,
            output_ptr=output + output_offset,
            batch_idx=b,
            n_channels=n_channels,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    """Return the optimized function"""
    def optimized_function(input_tensor):
        # Apply fused SiLU + spatial mean computation
        result = fused_silu_spatial_mean(input_tensor)
        return result
    
    return optimized_function