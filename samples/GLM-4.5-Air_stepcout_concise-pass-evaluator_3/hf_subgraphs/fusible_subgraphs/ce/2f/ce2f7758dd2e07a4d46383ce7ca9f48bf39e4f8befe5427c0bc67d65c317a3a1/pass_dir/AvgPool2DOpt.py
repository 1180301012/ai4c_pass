import torch
import triton
import triton.language as tl

def pattern(x_16):
    """
    Pattern matching optimized AvgPool2D operation
    x_16: input tensor to average pool
    """
    # Standardized pooling parameters from all graphs
    tmp_7 = torch.nn.functional.avg_pool2d(x_16, 2, 2, 0, True, False, None)
    return tmp_7

def replacement_args(x_16):
    return (x_16,)

# Simple and robust average pooling kernel for 2x2 pooling with stride 2
@triton.jit
def avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    pool_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total threads needed and check bounds
    total_threads = batch_size * channels * output_height * output_width
    if pid >= total_threads:
        return
    
    # Decompose thread ID into components
    remaining = pid
    batch_idx = remaining // (channels * output_height * output_width)
    remaining %= channels * output_height * output_width
    channel_idx = remaining // (output_height * output_width)
    remaining %= output_height * output_width
    h_out = remaining // output_width
    w_out = remaining % output_width
    
    # Calculate input window coordinates
    h_in_start = h_out * stride
    w_in_start = w_out * stride
    
    # Simple pooling - handle boundary conditions
    sum_val = 0.0
    count = 0.0
    
    # Use range for iteration (Triton compatible)
    for h_offset in range(2):
        h_in = h_in_start + h_offset
        for w_offset in range(2):
            w_in = w_in_start + w_offset
            
            # Check bounds
            if h_in < input_height and w_in < input_width:
                # Input layout: [batch, channels, height, width]
                input_offset = (batch_idx * channels + channel_idx) * input_height * input_width + h_in * input_width + w_in
                x_val = tl.load(x_ptr + input_offset)
                sum_val += x_val
                count += 1.0
    
    # Calculate average
    avg_val = sum_val / count if count > 0 else 0.0
    
    # Store result
    output_offset = (batch_idx * channels + channel_idx) * output_height * output_width + h_out * output_width + w_out
    tl.store(out_ptr + output_offset, avg_val)

# Simplified average pooling kernel - removing complex vectorization for better compatibility
# The basic kernel is optimized for 2x2 pooling which is the common case

@torch.fx.wrap 
def optimized_avg_pool2d(x_16):
    device = x_16.device
    x_16 = x_16.contiguous()
    
    # Get tensor shapes
    batch_size, channels, input_height, input_width = x_16.shape
    
    # Standard pooling parameters from all graphs
    pool_size = 2
    stride = 2
    padding = 0
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - pool_size) // stride + 1
    output_width = (input_width + 2 * padding - pool_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=device, dtype=x_16.dtype)
    
    # Total elements for grid calculation
    total_elements = batch_size * channels * output_height * output_width
    
    # Launch optimized kernel for 2x2 pooling
    # Use tuple grid specification as required by Triton
    grid = (total_elements,)
    avg_pool2d_kernel[grid](
        x_ptr=x_16,
        out_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        pool_size=pool_size,
        stride=stride,
        padding=padding,
    )
    
    return output

def replacement_func():
    return optimized_avg_pool2d