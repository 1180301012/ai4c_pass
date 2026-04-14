import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.3, False, True)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel: SiLU + Adaptive Avg Pool2d + Flatten + Dropout
@triton.jit
def fused_silu_pool_flatten_dropout_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    dropout_prob: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel in the batch
    pid = tl.program_id(0)
    
    # Calculate total elements per channel in input
    elements_per_channel = height * width
    total_elements = batch_size * channels
    
    # Create mask for valid channels
    mask = pid < channels
    
    if mask:
        # For each batch, compute sum over spatial dimensions for this channel
        batch_offset = pid * elements_per_channel
        x_channel_ptr = x_ptr + batch_offset
        
        # Sum all spatial elements for each batch
        sum_val = tl.zeros([batch_size], dtype=tl.float32)
        
        for h in range(height):
            for w in range(width):
                # Load all batch elements for this spatial position
                spatial_offset = h * width + w
                x_ptr_batch = x_channel_ptr + spatial_offset
                spatial_vals = tl.load(x_ptr_batch, mask=spatial_offset < total_elements, other=0.0)
                sum_val += spatial_vals
        
        # Convert to average
        avg_val = sum_val / elements_per_channel
        
        # Apply SiLU activation: x * sigmoid(x)
        sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
        silu_val = avg_val * sigmoid_val
        
        # Apply dropout
        dropout_mask = tl.random_like(silu_val) > dropout_prob
        output_val = silu_val * dropout_mask
        
        # Store to output (flatten removes spatial dimensions)
        out_offset = pid
        out_ptr_batch = out_ptr + out_offset
        tl.store(out_ptr_batch, output_val, mask=out_offset < total_elements)

@torch.fx.wrap
def fused_silu_pool_flatten_dropout(in_0):
    # Get input tensor properties
    batch_size, channels, height, width = in_0.shape
    dtype = in_0.dtype
    
    # Convert to float32 for computation if needed
    if dtype != torch.float32:
        in_0_f32 = in_0.to(torch.float32)
    else:
        in_0_f32 = in_0
    
    # Create output tensor
    out_shape = (batch_size, channels)
    output = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Calculate grid and block sizes
    total_channels = batch_size * channels
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (total_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with dropout probability 0.3
    fused_silu_pool_flatten_dropout_kernel[(num_programs,)](
        x_ptr=in_0_f32,
        out_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        dropout_prob=0.3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_silu_pool_flatten_dropout