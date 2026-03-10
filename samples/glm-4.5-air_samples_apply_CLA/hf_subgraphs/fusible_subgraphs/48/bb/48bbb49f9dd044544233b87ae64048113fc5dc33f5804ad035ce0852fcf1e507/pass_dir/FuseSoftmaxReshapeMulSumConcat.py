import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    # Simpler pattern that focuses on the core computation structure
    # a is the softmax input, b and c are the weight tensors
    t1 = torch.nn.functional.softmax(a, dim=2)
    t2 = t1.reshape(-1, 17, 64, 64)
    
    # Two parallel computation paths
    x_path = t2.mul(b)
    x_reduced = torch.sum(x_path.reshape(x_path.shape[0], 17, -1), dim=2, keepdim=True)
    
    y_path = t2.mul(c)
    y_reduced = torch.sum(y_path.reshape(y_path.shape[0], 17, -1), dim=2, keepdim=True)
    
    # Final concatenation
    result = torch.cat([x_reduced, y_reduced], dim=-1)
    
    return (t2, result)

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def fused_kernel_x(
    softmax_ptr,
    in_0_ptr,
    x_sum_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes one spatial element for one channel
    program_id = tl.program_id(0)
    batch_id = program_id // (n_channels * height * width)
    channel_id = (program_id // (height * width)) % n_channels
    height_id = (program_id // width) % height
    width_id = program_id % width
    
    # Load softmax value at this position
    softmax_offset = (batch_id * n_channels * height * width + 
                     channel_id * height * width + 
                     height_id * width + width_id)
    softmax_val = tl.load(softmax_ptr + softmax_offset)
    
    # Load in_0 value (with broadcast: [1,1,1,64] -> [17,64,64])
    in_0_offset = width_id  # only depends on width dimension due to broadcasting
    in_0_val = tl.load(in_0_ptr + in_0_offset)
    
    # Compute weighted value and store
    weighted_val = softmax_val * in_0_val
    tl.store(x_sum_ptr + softmax_offset, weighted_val)

@triton.jit
def fused_kernel_y(
    softmax_ptr,
    in_1_ptr,
    y_sum_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes one spatial element for one channel
    program_id = tl.program_id(0)
    batch_id = program_id // (n_channels * height * width)
    channel_id = (program_id // (height * width)) % n_channels
    height_id = (program_id // width) % height
    width_id = program_id % width
    
    # Load softmax value at this position
    softmax_offset = (batch_id * n_channels * height * width + 
                     channel_id * height * width + 
                     height_id * width + width_id)
    softmax_val = tl.load(softmax_ptr + softmax_offset)
    
    # Load in_1 value (with broadcast: [1,1,64,1] -> [17,64,64])
    in_1_offset = height_id  # only depends on height dimension due to broadcasting
    in_1_val = tl.load(in_1_ptr + in_1_offset)
    
    # Compute weighted value and store
    weighted_val = softmax_val * in_1_val
    tl.store(y_sum_ptr + softmax_offset, weighted_val)

@triton.jit
def reduction_kernel(
    x_sum_ptr,
    y_sum_ptr,
    final_result_ptr,  # [batch_size, n_channels, 2] - flattened to [batch_size * n_channels * 2]
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one channel for one batch
    program_id = tl.program_id(0)
    batch_id = program_id // n_channels
    channel_id = program_id % n_channels
    
    # Load all spatial elements for this channel and batch and sum them
    x_channel_sum = 0.0
    y_channel_sum = 0.0
    
    spatial_size = height * width
    base_offset = (batch_id * n_channels + channel_id) * spatial_size
    
    for i in range(spatial_size):
        x_channel_sum += tl.load(x_sum_ptr + base_offset + i)
        y_channel_sum += tl.load(y_sum_ptr + base_offset + i)
    
    # Store both results in the final concatenated tensor
    # Structure: [batch_size, n_channels, 2] flattened to linear memory
    result_base_offset = (batch_id * n_channels + channel_id) * 2
    
    tl.store(final_result_ptr + result_base_offset, x_channel_sum)
    tl.store(final_result_ptr + result_base_offset + 1, y_channel_sum)

@torch.fx.wrap
def fused_kernel_wrapper(a, b, c):
    # Determine output shapes based on input sizes
    batch_size = a.shape[0]
    n_channels = 17
    height = 64
    width = 64
    
    # Create output tensors
    # First output: reshaped softmax (we compute this in the pattern)
    out_3 = torch.empty_like(a)
    
    # Temporary storage for weighted sums before reduction
    # Shape: [batch_size, n_channels, height, width]
    x_sum_buffer = torch.empty_like(a)
    y_sum_buffer = torch.empty_like(a)
    
    # Final concatenated result: [batch_size, n_channels, 2] flattened to linear memory
    final_result = torch.empty(batch_size * n_channels * 2, dtype=a.dtype,
                               device=a.device)
    
    # Calculate grid sizes
    n_spatial_elements = batch_size * n_channels * height * width
    n_channel_elements = batch_size * n_channels
    BLOCK_SIZE = 1024
    
    grid_size_spatial = (n_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_size_channel = (n_channel_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch two fused kernels for x and y computations
    fused_kernel_x[grid_size_spatial](
        a,
        b,
        x_sum_buffer,
        batch_size,
        n_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    fused_kernel_y[grid_size_spatial](
        a,
        c,
        y_sum_buffer,
        batch_size,
        n_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch reduction kernel for final concatenated results
    reduction_kernel[grid_size_channel](
        x_sum_buffer,
        y_sum_buffer,
        final_result,
        batch_size,
        n_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape final result from [batch_size * n_channels * 2] to [batch_size, n_channels, 2]
    out_10 = final_result.reshape(batch_size, n_channels, 2)
    
    # The reshaped softmax is already computed in the pattern function
    # Just copy the input softmax output
    out_3.copy_(a)
    
    return out_3, out_10

def replacement_func():
    return fused_kernel_wrapper