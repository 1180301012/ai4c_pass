import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def correct_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size * channels:
        return
    
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Each program handles a block of spatial elements
    block_start = tl.program_id(1) * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < height * width
    
    # Calculate the starting index for this batch-channel pair
    # After sum(dim=1), we treat the tensor as [batch_size, channels, height, width]
    base_offset = (batch_idx * channels + channel_idx) * height * width
    
    # Load a block of spatial elements
    values = tl.load(input_ptr + base_offset + indices, mask=mask)
    
    # Sum the loaded values
    partial_sum = tl.sum(values)
    
    # Only the first block does the final computation and storage
    if block_start == 0:
        avg_value = partial_sum / (height * width)
        tl.store(output_ptr + pid, avg_value)

@torch.fx.wrap
def correct_sum_and_pool(input_tensor):
    original_shape = input_tensor.shape
    if len(original_shape) == 5:
        batch_size, orig_channels, height, width, depth = original_shape
        
        # After sum(dim=1), we have [batch_size, orig_channels, height, width]
        # The depth dimension gets summed over, so it disappears
        channels = orig_channels
    else:
        raise ValueError(f"Expected 5D tensor, got {original_shape}")
    
    output_size = batch_size * channels
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use 2D grid to handle large spatial dimensions
    spatial_elements = height * width
    BLOCK_SIZE = 256  # Power of 2
    num_blocks = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (output_size, num_blocks)
    
    correct_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # The result should be [batch_size, channels, 1, 1]
    output_reshaped = output.reshape(batch_size, channels, 1, 1)
    return output_reshaped

def replacement_func():
    return correct_sum_and_pool