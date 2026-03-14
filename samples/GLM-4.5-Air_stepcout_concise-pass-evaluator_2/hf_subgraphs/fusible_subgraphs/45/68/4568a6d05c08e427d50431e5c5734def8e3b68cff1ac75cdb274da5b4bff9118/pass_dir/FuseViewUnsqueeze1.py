import torch
import triton
import triton.language as tl

def pattern(relu_out):
    # Pattern matches: view + unsqueeze(1) sequence
    # relu_out comes from the previous ReLU operation
    tmp_1 = relu_out.view(relu_out.shape[0], relu_out.shape[1], -1)  # view combining spatial dims
    tmp_2 = tmp_1.unsqueeze(1)  # unsqueeze at position 1
    return tmp_2, relu_out  # return both unsqueezed tensor and original relu output

def replacement_args(relu_out):
    return (relu_out,)

@triton.jit
def fuse_view_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    # Output shape: [batch_size, 1, channels, spatial_size]
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Calculate output indices
    output_offset = (batch_idx * channels * spatial_size + 
                    channel_idx * spatial_size + 
                    spatial_idx)
    
    mask = (batch_idx < batch_size) & (channel_idx < channels) & (spatial_idx < spatial_size)
    
    # Calculate input offset - input has shape [batch_size, channels, spatial_size]
    input_offset = (batch_idx * channels * spatial_size + 
                   channel_idx * spatial_size + 
                   spatial_idx)
    
    # Load from input (flat spatial dimensions)
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output with added dimension at position 1
    tl.store(output_ptr + output_offset, val, mask=mask)

@torch.fx.wrap
def fuse_view_unsqueeze(relu_out):
    # Input shape: [batch_size, channels, height, width]
    # We need to compute: 
    # 1. view to [batch_size, channels, height*width] 
    # 2. unsqueeze(1) to [batch_size, 1, channels, height*width]
    
    batch_size, channels, height, width = relu_out.shape
    spatial_size = height * width
    
    # Compute output shape: [batch_size, 1, channels, spatial_size]
    output_shape = (batch_size, 1, channels, spatial_size)
    output = torch.empty(output_shape, dtype=relu_out.dtype, device=relu_out.device)
    
    # Block size for better GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_channels = channels
    num_spatial = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fuse_view_unsqueeze_kernel[(num_batches, num_channels, num_spatial)](
        relu_out,
        output,
        batch_size,
        channels,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, relu_out

def replacement_func():
    return fuse_view_unsqueeze