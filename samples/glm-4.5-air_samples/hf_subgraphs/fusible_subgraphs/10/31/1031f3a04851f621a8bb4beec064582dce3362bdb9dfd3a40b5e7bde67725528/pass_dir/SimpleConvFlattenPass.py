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
def simple_conv_flatten_kernel(
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
    # Program id
    pid = tl.program_id(0)
    
    # Calculate total number of output elements per batch and
    total_elements_per_batch = n_channels_out * height * width
    total_elements = n_batch * total_elements_per_batch
    
    if pid >= total_elements:
        return
    
    # Calculate batch index and element index within batch
    batch_idx = pid // total_elements_per_batch
    element_idx = pid % total_elements_per_batch
    
    # Calculate channel, row, and column from element index
    channel_idx = element_idx // (height * width)
    spatial_idx = element_idx % (height * width)
    row_idx = spatial_idx // width
    col_idx = spatial_idx % width
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform 1x1 convolution (dot product over input channels)
    for c_in in range(n_channels_in):
        # Input offset: (batch_idx, c_in, row_idx, col_idx)
        input_offset = (batch_idx * n_channels_in + c_in) * height * width + row_idx * width + col_idx
        input_val = tl.load(input_ptr + input_offset)
        
        # Weight offset: (channel_idx, c_in, 0, 0) 
        weight_offset = channel_idx * n_channels_in + c_in
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Multiply and accumulate
        acc += input_val * weight_val
    
    # Load bias and add to result
    bias_val = tl.load(bias_ptr + channel_idx)
    result = acc + bias_val
    
    # Store result
    output_offset = pid
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def simple_conv_flatten_wrapper(in_0, in_1, in_2):
    # Get input dimensions
    n_batch, n_channels_in, height, width = in_2.shape
    n_channels_out = in_0.shape[0]
    
    # Output shape: [n_batch, n_channels_out, height, width] -> flatten dim 2
    output_shape = (n_batch, n_channels_out * height * width)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Use simple blocking - each program handles one element
    total_elements = n_batch * n_channels_out * height * width
    block_size = 1
    
    # Launch grid with one program per output element
    grid = [total_elements]
    
    simple_conv_flatten_kernel[grid](
        in_2.data_ptr(),
        in_1.data_ptr(),
        in_0.data_ptr(),
        output.data_ptr(),
        n_batch,
        n_channels_in,
        height,
        width,
        n_channels_out,
        block_size,
    )
    
    # Reshape output to match original flattened format
    return output.view(n_batch, n_channels_out, height * width)

def replacement_func():
    return simple_conv_flatten_wrapper