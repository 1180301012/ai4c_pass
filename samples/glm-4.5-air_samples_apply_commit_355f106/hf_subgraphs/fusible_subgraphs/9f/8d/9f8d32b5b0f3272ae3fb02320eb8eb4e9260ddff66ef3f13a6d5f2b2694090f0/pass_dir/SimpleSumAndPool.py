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
def simple_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size * channels:
        return
    
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Load spatial elements for this batch-channel pair
    indices = tl.arange(0, height * width)
    mask = indices < height * width
    
    base_offset = (batch_idx * channels + channel_idx) * height * width
    values = tl.load(input_ptr + base_offset + indices, mask=mask)
    
    # Compute average
    avg_value = tl.sum(values) / (height * width)
    
    # Store result
    tl.store(output_ptr + pid, avg_value)

@torch.fx.wrap
def simple_sum_and_pool(input_tensor):
    original_shape = input_tensor.shape
    if len(original_shape) == 5:
        batch_size, orig_channels, height, width, depth = original_shape
        
        # We need to interpret the tensor as [batch, channels, height, width]
        # where channels = orig_channels * depth (after sum operation)
        channels = orig_channels
    else:
        raise ValueError(f"Expected 5D tensor, got {original_shape}")
    
    output_size = batch_size * channels
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    grid = lambda meta: (output_size,)
    
    simple_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    
    # Reshape to [batch_size, channels, 1, 1]
    output_reshaped = output.reshape(batch_size, channels, 1, 1)
    return output_reshaped

def replacement_func():
    return simple_sum_and_pool