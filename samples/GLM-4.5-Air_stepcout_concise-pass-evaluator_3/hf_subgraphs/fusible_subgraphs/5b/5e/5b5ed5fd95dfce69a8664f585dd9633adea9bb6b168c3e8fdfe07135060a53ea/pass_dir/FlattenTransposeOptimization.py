import torch
import triton
import triton.language as tl

def pattern(input_feat):
    # The sequence: flatten(2) followed by transpose(1, 2)
    tmp_6 = input_feat.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(input_feat):
    return (input_feat,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output [B, H*W, C]
    pid = tl.program_id(0)
    
    # Calculate coordinates in output tensor
    b = pid // (channels * height * width)
    hw_pos = (pid // channels) % (height * width)  # flattened spatial position
    c = pid % channels
    
    if b >= batch_size:
        return
    if hw_pos >= height * width:
        return
    if c >= channels:
        return
    
    # Calculate original coordinates from flattened spatial position
    h = hw_pos // width
    w = hw_pos % width
    
    # Calculate input index [B, C, H, W]
    input_idx = b * channels * height * width + c * height * width + h * width + w
    
    # Calculate output index [B, H*W, C]  
    output_idx = b * channels * height * width + hw_pos * channels + c
    
    # Load from input and store to output (this effectuates flatten + transpose)
    val = tl.load(input_ptr + input_idx)
    tl.store(output_ptr + output_idx, val)

@torch.fx.wrap
def flatten_transpose_optimized(input_feat):
    batch_size, channels, height, width = input_feat.shape
    
    output = torch.empty((batch_size, height * width, channels), dtype=input_feat.dtype, device=input_feat.device)
    
    # Calculate total number of elements
    total_elements = batch_size * channels * height * width
    
    # Choose block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    flatten_transpose_kernel[(num_programs,)](
        input_ptr=input_feat,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return flatten_transpose_optimized