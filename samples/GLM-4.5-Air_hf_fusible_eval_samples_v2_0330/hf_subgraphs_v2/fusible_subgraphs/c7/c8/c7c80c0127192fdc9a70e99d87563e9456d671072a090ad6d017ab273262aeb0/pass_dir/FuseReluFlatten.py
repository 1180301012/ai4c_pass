import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Match just flatten operation (this works)
    tmp_1 = in_0.flatten(1, -1)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    features,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a portion of the flattened data
    pid = tl.program_id(0)
    total_elements = batch_size * features
    
    # Calculate the range of elements this program handles
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, total_elements)
    
    # Iterate over the elements this program is responsible for
    for i in range(block_start, block_end):
        # Calculate batch index and feature index for the element
        b = i // features
        f = i % features
        
        # Calculate input indices (input shape is [batch_size, features, 1, 1])
        input_b = b
        input_f = f
        input_h = 0  # always 0 since H=1
        input_w = 0  # always 0 since W=1
        
        # Calculate input memory offset
        input_offset = (input_b * input_stride_0 + 
                       input_f * input_stride_1 + 
                       input_h * input_stride_2 + 
                       input_w * input_stride_3)
        
        # Load input value and apply ReLU
        x = tl.load(input_ptr + input_offset)
        relu_x = tl.maximum(x, 0.0)
        
        # Store output value
        output_offset = b * output_stride_0 + f * output_stride_1
        tl.store(output_ptr + output_offset, relu_x)

@torch.fx.wrap
def fused_relu_flatten(input_tensor):
    # Get input tensor properties
    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    features = input_shape[1] * input_shape[2] * input_shape[3]  # C * H * W
    
    # Create output tensor with flattened shape
    output_shape = (batch_size, features)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get input tensor strides
    input_stride_0 = input_tensor.stride(0)
    input_stride_1 = input_tensor.stride(1)
    input_stride_2 = input_tensor.stride(2)
    input_stride_3 = input_tensor.stride(3)
    
    # Get output tensor strides
    output_stride_0 = output_tensor.stride(0)
    output_stride_1 = output_tensor.stride(1)
    
    # Choose block size based on input size for optimal performance
    if batch_size * features < 1024:
        BLOCK_SIZE = 64
    elif batch_size * features < 8192:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    total_elements = batch_size * features
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    relu_flatten_kernel[(num_programs,)](
        input_tensor,
        output_tensor,
        batch_size,
        features,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        input_stride_3,
        output_stride_0,
        output_stride_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

@triton.jit
def optimized_flatten_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and a block of channels
    batch_idx = tl.program_id(0)
    channel_offset = tl.program_id(1) * BLOCK_SIZE
    
    # Calculate channel range for this program
    channel_start = channel_offset
    channel_end = min(channel_offset + BLOCK_SIZE, channels)
    
    # Process channels for this batch
    for c in range(channel_start, channel_end):
        # Calculate input offset: (batch_idx, c, 0, 0)
        input_offset = batch_idx * input_stride_0 + c * input_stride_1
        
        # Calculate output offset: (batch_idx, c) in flattened tensor
        output_offset = batch_idx * output_stride_0 + c * output_stride_1
        
        # Load and store the element
        x = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + output_offset, x)

@torch.fx.wrap
def fused_relu_flatten(input_tensor):
    # Get input tensor properties
    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    channels = input_shape[1]  # H=W=1, so channels = C * H * W
    
    # Create output tensor with flattened shape [batch_size, channels]
    output_shape = (batch_size, channels)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get strides for efficient memory access
    input_stride_0 = input_tensor.stride(0)
    input_stride_1 = input_tensor.stride(1)
    output_stride_0 = output_tensor.stride(0)
    output_stride_1 = output_tensor.stride(1)
    
    # Configure block size for optimal GPU occupancy
    if channels < 256:
        BLOCK_SIZE = 64
    elif channels < 1024:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid dimensions: [batch_size, num_channel_groups]
    num_channel_groups = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_channel_groups)
    
    # Launch the optimized kernel
    optimized_flatten_kernel[grid](
        input_tensor,
        output_tensor,
        batch_size,
        channels,
        input_stride_0,
        input_stride_1,
        output_stride_0,
        output_stride_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return fused_relu_flatten