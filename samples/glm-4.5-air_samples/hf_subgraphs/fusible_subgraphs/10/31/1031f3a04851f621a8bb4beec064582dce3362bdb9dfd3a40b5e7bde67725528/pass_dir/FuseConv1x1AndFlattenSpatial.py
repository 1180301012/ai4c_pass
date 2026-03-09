import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_1x1_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_batch,
    n_channels_in,
    height,
    width,
    n_channels_out,
    BLOCK_SIZE: tl.constexpr
):
    # Program id with 1D grid
    pid = tl.program_id(0)
    
    # Calculate grid size and return if out of bounds
    total_elements = n_batch * n_channels_out * height * width
    if pid >= total_elements:
        return
    
    # Extract coordinates from 1D index
    spatial_idx = pid % (height * width)
    batch_channel_idx = pid // (height * width)
    batch_idx = batch_channel_idx // n_channels_out
    channel_idx = batch_channel_idx % n_channels_out
    
    # Calculate output offset
    output_offset = batch_channel_idx * height * width + spatial_idx
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Process dot product over input channels
    acc = 0.0
    for k in range(0, n_channels_in):
        # Calculate input offset for this channel and spatial position
        input_offset = (batch_idx * n_channels_in + k) * height * width + spatial_idx
        
        # Calculate weight offset for this channel pair
        weight_offset = channel_idx * n_channels_in + k
        
        # Load input and weight values
        input_val = tl.load(input_ptr + input_offset)
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Multiply and accumulate
        acc += input_val * weight_val
    
    # Store result with bias
    tl.store(output_ptr + output_offset, acc + bias_val)

@torch.fx.wrap
def fused_conv_1x1_flatten_wrapper(in_0, in_1, in_2):
    # Get input dimensions
    n_batch, n_channels_in, height, width = in_2.shape
    n_channels_out = in_0.shape[0]
    
    # Output shape: [n_batch, n_channels_out, height, width] -> flatten dim 2: [n_batch, n_channels_out, height * width]
    output_shape = (n_batch, n_channels_out, height * width)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # For this simple kernel, use one program per output element for maximum simplicity
    # Each program handles one element from the flattened output tensor
    total_elements = n_batch * n_channels_out * height * width
    
    fused_conv_1x1_flatten_kernel[(total_elements,)](
        in_2.data_ptr(),
        in_1.data_ptr(),
        in_0.data_ptr(),
        output.data_ptr(),
        n_batch,
        n_channels_in,
        height,
        width,
        n_channels_out,
        1,  # BLOCK_SIZE not used in kernel
    )
    
    return output

def replacement_func():
    return fused_conv_1x1_flatten_wrapper