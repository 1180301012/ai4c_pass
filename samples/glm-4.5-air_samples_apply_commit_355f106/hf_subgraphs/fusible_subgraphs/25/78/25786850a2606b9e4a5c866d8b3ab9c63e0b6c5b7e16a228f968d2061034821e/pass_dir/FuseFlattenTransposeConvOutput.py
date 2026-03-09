import torch
import triton
import triton.language as tl

def pattern(conv_out):
    # Matches: flatten(2) -> transpose(1, 2)
    tmp_6 = conv_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(conv_out):
    return (conv_out,)

@triton.jit
def fused_flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len * channels
    
    # Calculate offsets for parallel processing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    batch = offsets // (seq_len * channels)
    rem = offsets % (seq_len * channels)
    position = rem // channels
    channel = rem % channels
    
    # Convert position (0 to seq_len-1) to height, width coordinates
    h = position // width
    w = position % width
    
    # Coalesced memory access pattern for input
    input_offset = batch * (channels * height * width) + channel * (height * width) + h * width + w
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Coalesced memory access pattern for output  
    output_offset = batch * (seq_len * channels) + position * channels + channel
    tl.store(output_ptr + output_offset, x, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose_gpu(conv_out):
    # Input: [batch, channels, height, width] = [1, 768, 224, 224]
    # Expected output: [batch, seq_len, features] = [1, 224*224, 768] = [1, 50176, 768]
    
    batch_size, channels, height, width = conv_out.shape
    seq_len = height * width  # 224 * 224 = 50176
    
    # Create output with correct shape: [batch, seq_len, features]
    output = torch.empty(batch_size, seq_len, channels, dtype=conv_out.dtype, device=conv_out.device)
    
    N = batch_size * seq_len * channels
    # Use larger block size for large tensor to improve GPU utilization
    BLOCK_SIZE = 4096  # Optimal block size for this workload (from testing)
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_flatten_transpose_kernel[(num_programs,)](
        input_ptr=conv_out,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_flatten_transpose_gpu