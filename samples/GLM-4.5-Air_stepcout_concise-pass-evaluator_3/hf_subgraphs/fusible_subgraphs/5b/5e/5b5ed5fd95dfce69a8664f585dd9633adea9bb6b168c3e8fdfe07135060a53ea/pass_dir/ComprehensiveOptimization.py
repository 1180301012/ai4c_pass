import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Match the sequence: flatten(2) followed by transpose(1, 2)
    # This represents a common reshape operation in transformer networks
    tmp_6 = input_tensor.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one element in the output [B, H*W, C]
    if pid >= batch_size * height * width * channels:
        return
        
    # Calculate coordinates in output layout [B, H*W, C]
    b = pid // (height * width * channels)
    remainder = pid % (height * width * channels)
    hw_pos = remainder // channels
    c = remainder % channels
    
    # Calculate coordinates in input layout [B, C, H, W]
    h = hw_pos // width
    w = hw_pos % width
    
    # Calculate linear indices
    input_idx = b * channels * height * width + c * height * width + h * width + w
    output_idx = b * (height * width) * channels + hw_pos * channels + c
    
    # Direct memory-to-memory data movement
    val = tl.load(input_ptr + input_idx)
    tl.store(output_ptr + output_idx, val)

@torch.fx.wrap  
def optimized_reshape(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    
    output = torch.empty((batch_size, height * width, channels), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    total_elements = batch_size * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_reshape