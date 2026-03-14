import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match: ReLU + Add + AdaptiveAvgPool2d
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_relu_add_avgpool_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for ReLU + Add + AdaptiveAvgPool2d.
    Each program handles one (batch, channel) position.
    """
    # Program ID represents (batch * channels + channel)
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Base offset for this (batch, channel) position
    base_offset = pid * spatial_size
    
    # Accumulator for the sum
    sum_val = 0.0
    
    # Process all spatial elements in blocks
    for block_start in range(0, spatial_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load input data
        in_0_data = tl.load(in_0_ptr + base_offset + offsets, mask=mask, other=0.0)
        in_1_data = tl.load(in_1_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        # Apply ReLU to in_1
        relu_result = tl.maximum(in_1_data, 0.0)
        
        # Add in_0
        add_result = relu_result + in_0_data
        
        # Accumulate sum
        sum_val += tl.sum(add_result)
    
    # Compute average
    avg_val = sum_val / spatial_size
    
    # Store result
    tl.store(out_ptr + pid, avg_val)


@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    """
    Wrapper function for the fused kernel.
    """
    batch_size, channels, height, width = in_0.shape
    spatial_size = height * width
    
    # Output shape: [batch_size, channels, 1, 1]
    out = torch.empty((batch_size, channels, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Grid: one program per (batch, channel)
    grid = (batch_size * channels,)
    
    fused_relu_add_avgpool_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
    )
    
    return out


def replacement_func():
    return fused_relu_add_avgpool