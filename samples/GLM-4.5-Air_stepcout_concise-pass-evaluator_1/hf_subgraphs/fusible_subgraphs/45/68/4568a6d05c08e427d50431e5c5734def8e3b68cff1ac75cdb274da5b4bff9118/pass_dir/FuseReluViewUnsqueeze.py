import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(in_0.size(0), in_0.size(1), in_0.size(2) * in_0.size(3))
    tmp_2 = tmp_1.unsqueeze(1)
    return (tmp_2, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_view_unsqueeze_kernel(
    input_ptr,
    relu_out_ptr,
    final_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid setup based on total elements
    total_elements = batch_size * channels * height * width
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_data = tl.maximum(input_data, 0.0)
    
    # Store ReLU result
    tl.store(relu_out_ptr + offsets, relu_data, mask=mask)
    
    # For the final output: unsqueeze(1) means we add dimension at position 1
    # The output shape is [batch_size, 1, channels, height*width]
    # We need to write to a different memory layout
    
    # Calculate output indices for unsqueeze operation
    # Original flattened indices: idx = b * (C * H * W) + c * (H * W) + h * W + w
    # Output indices: [b, 1, c, h * W + w] -> needs different layout
    
    # We need to handle the reshape and unsqueeze in this kernel
    # Each thread handles multiple elements
    
    # Number of elements per channel
    elements_per_channel = height * width
    total_channels = batch_size * channels
    elements_per_batch = channels * elements_per_channel
    
    # Thread index within block
    thread_idx = offsets % BLOCK_SIZE
    
    # Find which batch/channel this thread handles
    linear_idx = offsets
    batch_idx = linear_idx // elements_per_batch
    remaining = linear_idx % elements_per_channel
    channel_idx = remaining // elements_per_channel
    element_idx = remaining % elements_per_channel
    
    # For the output with unsqueeze: [batch_size, 1, channels, height*width]
    # The output layout has an extra dimension, so we need special handling
    # We'll create a larger output tensor space
    
    # Simplified approach: let's handle the reshape first, then unsqueeze can be handled separately
    # For now, focus on the core fusion of ReLU + reshape



@triton.jit
def fused_relu_view_unsqueeze_kernel(
    input_ptr,
    relu_out_ptr,
    final_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of elements in input tensor
    total_elements = batch_size * channels * height * width
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    relu_data = tl.maximum(input_data, 0.0)
    
    # Store ReLU result (maintains original shape [B, C, H, W])
    tl.store(relu_out_ptr + offsets, relu_data, mask=mask)
    
    # For the final output with unsqueeze(1): [B, 1, C, H*W]
    # This is a reshape operation that just reinterprets the memory layout
    # The total number of elements remains the same: B*C*H*W
    tl.store(final_out_ptr + offsets, relu_data, mask=mask)

@torch.fx.wrap
def fused_relu_view_unsqueeze(in_0):
    input_shape = in_0.shape
    batch_size, channels, height, width = input_shape
    
    # Output shapes
    relu_out_shape = input_shape  # Same as input [B, C, H, W]
    final_out_shape = (batch_size, 1, channels, height * width)  # With unsqueeze
    
    # Allocate output tensors
    relu_out = torch.empty(relu_out_shape, dtype=in_0.dtype, device=in_0.device)
    final_out = torch.empty(final_out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch Triton kernel to fuse operations
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024  # Optimized block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_view_unsqueeze_kernel[(num_programs,)](
        input_ptr=in_0,
        relu_out_ptr=relu_out,
        final_out_ptr=final_out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return final_out, relu_out

def replacement_func():
    return fused_relu_view_unsqueeze