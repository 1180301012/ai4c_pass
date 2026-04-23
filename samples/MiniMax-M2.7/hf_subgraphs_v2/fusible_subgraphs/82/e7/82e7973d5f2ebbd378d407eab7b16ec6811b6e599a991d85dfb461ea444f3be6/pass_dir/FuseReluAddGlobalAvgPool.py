"""
Fused kernel for: relu(in_1) + in_0 followed by adaptive_avg_pool2d(output_size=1)
This fuses ReLU + Element-wise Add + Global Average Pooling into a single GPU kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_add_global_avgpool_kernel(
    in_0_ptr,        # Base pointer for input 0 (residual)
    in_1_ptr,        # Base pointer for input 1 (relu input)
    out_ptr,         # Output pointer [batch, channels, 1, 1]
    batch_strides,   # Strides for in_0: [batch_stride, channel_stride, h_stride, w_stride]
    n_elements,      # Total number of elements in in_0 (batch * channels * h * w)
    n_channels,      # Number of channels
    spatial_size,    # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. out = relu(in_1) + in_0
    2. global_avg_pool2d(out) -> mean over spatial dimensions
    
    For each (batch, channel), we:
    1. Compute relu(in_1) + in_0 for all spatial positions
    2. Sum all spatial values
    3. Divide by spatial_size to get average
    """
    pid = tl.program_id(0)
    
    # Each program handles one (batch, channel) pair
    # Total programs = batch * channels
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Compute pointers for this batch, channel
    # Shape: [batch, channels, h, w]
    # We need to iterate over all spatial positions
    
    # Accumulator for sum
    sum_acc = tl.zeros(1, dtype=tl.float32)
    
    # Iterate over spatial dimensions in blocks
    for spatial_start in range(0, spatial_size, BLOCK_SIZE):
        spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
        mask = spatial_offsets < spatial_size
        
        # Compute flat index for in_0
        # stride_b * batch + stride_c * channel + stride_h * h + stride_w * w
        # For the flattened access, we use: base + batch*batch_stride + channel*channel_stride + offset
        batch_stride = batch_strides[0]
        channel_stride = batch_strides[1]
        
        # Flattened index for this spatial position
        flat_idx = batch_idx * batch_stride + channel_idx * channel_stride + spatial_offsets
        
        # Load in_0 (residual) and in_1
        in_0_val = tl.load(in_0_ptr + flat_idx, mask=mask, other=0.0)
        in_1_val = tl.load(in_1_ptr + flat_idx, mask=mask, other=0.0)
        
        # Apply ReLU to in_1 and add to in_0
        relu_in_1 = tl.maximum(in_1_val, 0.0)
        added = in_0_val + relu_in_1
        
        # Accumulate sum
        sum_acc += tl.sum(added, axis=0)
    
    # Divide by spatial size to get average
    avg = sum_acc / tl.cast(spatial_size, tl.float32)
    
    # Compute output flat index: batch * channels (output is [batch, channels, 1, 1])
    out_idx = batch_idx * n_channels + channel_idx
    
    # Store result
    tl.store(out_ptr + out_idx, avg)


@torch.fx.wrap
def fused_relu_add_global_avgpool_dispatcher(in_0, in_1, route=""):
    """
    Dispatcher function that handles the fused relu + add + global avg pool operation.
    The route parameter allows for different pass configurations while sharing this function.
    """
    batch, channels, h, w = in_0.shape
    spatial_size = h * w
    n_elements = batch * channels * h * w
    
    # Allocate output tensor [batch, channels, 1, 1]
    out = torch.empty((batch, channels, 1, 1), device=in_0.device, dtype=in_0.dtype)
    
    # Get strides for in_0/in_1 (they should have same shape and strides)
    batch_stride = in_0.stride(0)
    channel_stride = in_0.stride(1)
    
    # BLOCK_SIZE for iterating over spatial dimensions
    # Use 1024 as default, which can cover up to 1024 spatial elements
    BLOCK_SIZE = 1024
    
    # Grid: one program per (batch, channel) pair
    num_programs = batch * channels
    
    # Launch kernel
    fused_relu_add_global_avgpool_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_strides=[batch_stride, channel_stride, 0, 0],  # Only batch and channel strides needed
        n_elements=n_elements,
        n_channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: relu(in_1) + in_0 followed by adaptive_avg_pool2d
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "FuseReluAddGlobalAvgPool")


def replacement_func():
    return fused_relu_add_global_avgpool_dispatcher