import torch
import triton
import triton.language as tl


@triton.jit
def fused_silu_global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_channels,
    n_batch,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. SiLU activation: x * sigmoid(x)
    2. Global average pooling over spatial dimensions
    3. Output flattened to (batch, channels)
    
    This fuses adaptive_avg_pool2d + flatten into a single operation.
    """
    # Each program handles one output element [batch_idx, channel_idx]
    # Total programs = n_batch * n_channels
    pid = tl.program_id(0)
    
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Calculate offsets for loading
    # Input is laid out as [batch, channels, H, W]
    # But we can view it as [batch * channels, H * W] for contiguous access
    # However, PyTorch's default memory layout is [batch, channels, H, W] (NCHW)
    
    # For NCHW layout: offset = batch * channels * spatial + channel * spatial + spatial_idx
    base_input_offset = batch_idx * n_channels * spatial_size + channel_idx
    
    # Accumulator for sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over spatial dimensions
    for spatial_idx in range(spatial_size):
        input_offset = base_input_offset + spatial_idx
        
        # Load data
        x = tl.load(input_ptr + input_offset)
        
        # Compute SiLU: x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-x))
        silu_out = x * sig
        
        # Accumulate
        acc = acc + silu_out
    
    # Compute mean
    mean = acc / spatial_size
    
    # Store output [batch, channels]
    output_offset = batch_idx * n_channels + channel_idx
    tl.store(output_ptr + output_offset, mean)


@torch.fx.wrap
def fused_silu_global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
    tmp_0 = torch.nn.functional.silu(x, inplace=True)  # inplace on input
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    
    Returns pooled and flattened result.
    """
    # Input shape: [batch, channels, H, W]
    batch, channels, H, W = x.shape
    spatial_size = H * W
    
    # Output shape: [batch, channels]
    output = torch.empty((batch, channels), dtype=x.dtype, device=x.device)
    
    # Launch kernel - one program per output element [batch, channel]
    num_programs = batch * channels
    
    # Choose block size based on spatial size
    BLOCK_SIZE = 128
    
    fused_silu_global_avg_pool_kernel[(num_programs,)](
        x,
        output,
        batch * channels,
        channels,
        batch,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0):
    """
    Match the pattern: silu + adaptive_avg_pool2d + flatten
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_silu_global_avg_pool