import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_0 = None
    in_0 /= tmp_1
    tmp_2 = in_0
    tmp_1 = None
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a specific channel and spatial location
    channel = tl.program_id(0)
    spatial_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = spatial_idx < width
    
    # Load channel data: shape (1, height, width) -> flatten spatial dimensions
    stride_ch = height * width
    input_ptr_ch = input_ptr + channel * stride_ch
    spatial_data = tl.load(input_ptr_ch + spatial_idx, mask=mask, other=0.0)
    
    # Calculate sum across spatial dimension for this location
    sum_val = tl.sum(spatial_data)
    
    # Normalize by dividing each element by the sum
    normalized_data = spatial_data / (sum_val + 1e-6)  # Add small epsilon for stability
    
    # Store result
    output_ptr_ch = output_ptr + channel * stride_ch
    tl.store(output_ptr_ch + spatial_idx, normalized_data, mask=mask)

@torch.fx.wrap  
def fused_normalize(input_tensor):
    # Input shape: [1, channels, height, width]
    batch_size, channels, height, width = input_tensor.shape
    
    # Reshape to combine batch and dimensions, process as 1D
    total_elements = height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    # Launch kernel for each channel
    normalize_kernel[(channels, num_programs)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_normalize