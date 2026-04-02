import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Simple Conv2D pattern test"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def simple_conv2d_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Highly optimized 1x1 Conv2D kernel with parallel processing"""
    pid = tl.program_id(0)
    
    # Handle multiple batches per program for better GPU utilization
    batch_idx = pid // ((height * width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    linear_pos = pid % ((height * width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    if batch_idx >= batch_size:
        return
        
    # Process BLOCK_SIZE positions per program
    pos_start = linear_pos * BLOCK_SIZE
    pos_end = min(pos_start + BLOCK_SIZE, height * width)
    
    # Load bias once
    bias_val = tl.load(bias_ptr)
    
    # Process each position in the block
    for pos in range(pos_start, pos_end):
        result = bias_val
        
        # Compute dot product: sum(weight[c] * input[batch, c, pos])
        for c in range(in_channels):
            # Sequential channel access - better memory locality for weight
            weight_c = tl.load(weight_ptr + c)
            
            # Compute input address for this batch, channel, and position
            # input shape: [batch, channels, height, width]
            # Linearized: batch * C * H * W + c * H * W + pos
            input_offset = (batch_idx * in_channels * height * width + 
                          c * height * width + pos)
            input_val = tl.load(input_ptr + input_offset)
            
            result += weight_c * input_val
        
        # Store result
        output_offset = (batch_idx * height * width + pos)
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def simple_conv2d(bias, weight, input_tensor):
    """Simple Conv2D wrapper"""
    batch_size, in_channels, height, width = input_tensor.shape
    
    output = torch.empty((batch_size, 1, height, width), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Calculate optimal grid size for better GPU utilization
    total_positions = height * width
    BLOCK_SIZE = 64  # Process 64 positions per program
    programs_per_batch = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_programs = batch_size * programs_per_batch
    grid_size = (total_programs,)
    
    simple_conv2d_kernel[grid_size](
        bias,
        weight,
        input_tensor,
        output,
        batch_size,
        in_channels,
        height,
        width,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_conv2d