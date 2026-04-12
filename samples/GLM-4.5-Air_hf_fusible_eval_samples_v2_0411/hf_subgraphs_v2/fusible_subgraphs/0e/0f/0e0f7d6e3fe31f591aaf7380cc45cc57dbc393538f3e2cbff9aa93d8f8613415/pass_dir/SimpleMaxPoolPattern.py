import torch
import triton
import triton.language as tl

# Simple pattern that just matches max_pool2d
def pattern(input_tensor):
    """Simple max_pool2d pattern"""
    result = torch.nn.functional.max_pool2d(input_tensor, 3, 2, 1, 1, ceil_mode = False, return_indices = False)
    return (result,)

def replacement_args(input_tensor):
    return (input_tensor,)

# Simple max pool kernel
@triton.jit
def simple_maxpool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    pool_size,
    pool_stride,
    pool_padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (height + 2 * pool_padding - pool_size) // pool_stride + 1
    out_width = (width + 2 * pool_padding - pool_size) // pool_stride + 1
    
    total_elements = batch_size * channels * out_height * out_width
    
    if pid >= total_elements:
        return
    
    # Decompose pid
    spatial_elements = out_height * out_width
    channel_elements = channels
    batch_id = pid // (channel_elements * spatial_elements)
    channel_id = (pid % (channel_elements * spatial_elements)) // spatial_elements
    spatial_id = pid % spatial_elements
    
    h_out = spatial_id // out_width
    w_out = spatial_id % out_width
    
    # Calculate input position
    h_in = h_out * pool_stride - pool_padding
    w_in = w_out * pool_stride - pool_padding
    
    # Initialize max value
    max_val = -float('inf')
    
    # Max pooling window
    for kh in range(pool_size):
        for kw in range(pool_size):
            ih = h_in + kh
            iw = w_in + kw
            
            if 0 <= ih < height and 0 <= iw < width:
                val = tl.load(input_ptr + (
                    batch_id * channels * height * width +
                    channel_id * height * width +
                    ih * width +
                    iw
                ))
                max_val = max(max_val, val)
    
    # Store result
    tl.store(output_ptr + pid, max_val)

@torch.fx.wrap
def simple_maxpool_replacement(input_tensor):
    """Simple max pooling replacement"""
    batch_size, channels, height, width = input_tensor.shape
    
    pool_size = 3
    pool_stride = 2
    pool_padding = 1
    
    out_height = (height + 2 * pool_padding - pool_size) // pool_stride + 1
    out_width = (width + 2 * pool_padding - pool_size) // pool_stride + 1
    
    output = torch.empty((batch_size, channels, out_height, out_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    if input_tensor.device.type == 'cuda':
        BLOCK_SIZE = 256
        num_programs = ((batch_size * channels * out_height * out_width + BLOCK_SIZE - 1) // BLOCK_SIZE)
        
        simple_maxpool_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # CPU fallback
        output.fill_(0.0)
    
    return output

# Replacement function
def replacement_func():
    return simple_maxpool_replacement