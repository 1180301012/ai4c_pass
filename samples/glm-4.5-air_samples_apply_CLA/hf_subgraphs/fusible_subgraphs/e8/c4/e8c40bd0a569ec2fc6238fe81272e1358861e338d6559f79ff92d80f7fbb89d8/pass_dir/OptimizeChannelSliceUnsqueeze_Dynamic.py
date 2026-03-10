import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    # Match the slice and unsqueeze operations
    tmp_1 = x[slice(None, None, None), 0]
    tmp_2 = torch.unsqueeze(tmp_1, 1)
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for input shape [1, C, H, W] - dynamic sizing
@triton.jit
def channel_extract_kernel_dynamic(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program processes one element in the batch
    pid_batch = tl.program_id(0)
    pid_height = tl.program_id(1)
    pid_width = tl.program_id(2)
    
    # Compute offsets - extract first channel (index 0)
    input_offset = pid_batch * channels * height * width + 0 * height * width + pid_height * width + pid_width
    output_offset = pid_batch * 1 * height * width + 0 * height * width + pid_height * width + pid_width
    
    # Load input element (first channel only)
    input_val = tl.load(input_ptr + input_offset)
    # Store output element
    tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def channel_extract_wrapper_dynamic(input_tensor):
    """Optimized kernel for extracting first channel from any [1, C, H, W] input"""
    batch_size, channels, height, width = input_tensor.shape
    assert batch_size == 1, f"Only batch size 1 supported, got {batch_size}"
    
    # Create output tensor [1, 1, H, W]
    output = torch.empty((1, 1, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = lambda meta: (
        meta['batch_size'],
        meta['height'], 
        meta['width']
    )
    
    channel_extract_kernel_dynamic[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    
    return output

# Replacement function
def replacement_func():
    return channel_extract_wrapper_dynamic