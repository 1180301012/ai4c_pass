import torch
import triton
import triton.language as tl


def pattern(x):
    # Simpler pattern - just match mean + view
    m = x.mean(dim=2, keepdim=True)
    out = m.view(1, 1, -1)
    return m, out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['width'],
)
@triton.jit
def mean_view_kernel(
    input_ptr,
    output_ptr,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a channel
    pid = tl.program_id(0)
    
    if pid >= num_channels:
        return
    
    # Channel offset
    channel_offset = pid * height * width
    
    # Load and compute sum for this channel
    sum_val = 0.0
    num_elements_per_channel = height * width
    
    # Process in blocks
    for block_start in range(0, num_elements_per_channel, BLOCK_SIZE):
        block_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < num_elements_per_channel
        
        # Global offsets for this channel
        offsets = channel_offset + block_offsets
        
        # Load input values
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Add to sum
        sum_val += tl.sum(tl.where(mask, x, 0.0), axis=0)
    
    # Compute mean
    mean_val = sum_val / num_elements_per_channel
    
    # Store mean (as a scalar for this channel)
    tl.store(output_ptr + pid, mean_val)


@torch.fx.wrap
def mean_view_kernel_wrapper(x):
    # Get input shape
    batch_size, num_channels, height, width = x.shape
    
    # Create mean output tensor
    mean_tensor = torch.empty((batch_size, num_channels), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one program per channel
    grid = (num_channels,)
    
    mean_view_kernel[grid](
        x,
        mean_tensor,
        num_channels,
        height,
        width,
    )
    
    # View to (1, 1, -1)
    out = mean_tensor.view(1, 1, -1)
    
    return mean_tensor, out


def replacement_func():
    return mean_view_kernel_wrapper