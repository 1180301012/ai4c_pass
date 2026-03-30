import torch
import triton
import triton.language as tl

def pattern(in_3):
    # Optimize avg_pool1d operation
    tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    return tmp_5

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def optimized_avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    batch_size, in_channels, input_length,
    kernel_size, stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output length
    output_length = (input_length + 2 * 0 - 1 * (kernel_size - 1) - 1) // stride + 1
    
    # Each program handles one feature dimension in one batch
    batch_idx = tl.program_id(0)
    ch_idx = tl.program_id(1)
    
    # Only process if within bounds
    if batch_idx >= batch_size or ch_idx >= in_channels:
        return
    
    # Program offset within the channel
    pid = tl.program_id(2)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length
    
    # Initialize pools for each output position
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pool_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Compute average pooling for this batch and channel
    for k in range(kernel_size):
        # Calculate input position
        in_seq_offsets = offsets * stride + k
        
        # Create mask for valid positions
        in_mask = (in_seq_offsets >= 0) & (in_seq_offsets < input_length)
        
        # Load input values (only if valid)
        input_vals = tl.load(input_ptr + batch_idx * in_channels * input_length + 
                            ch_idx * input_length + in_seq_offsets,
                            mask=in_mask, other=0.0)
        
        # Accumulate and count valid positions
        pool_sum += input_vals
        pool_count += in_mask
    
    # Compute average (avoid division by zero)
    pool_avg = pool_sum / tl.maximum(pool_count, 1)
    
    # Store result
    tl.store(output_ptr + batch_idx * in_channels * output_length + 
             ch_idx * output_length + offsets, pool_avg, mask=mask)

@torch.fx.wrap
def optimized_avg_pool1d(in_3):
    # Get input shape
    batch_size, in_channels, input_length = in_3.shape
    
    # Get pooling parameters
    kernel_size = 2
    stride = 2
    padding = 0
    
    # Calculate output length
    output_length = (input_length + 2 * padding - 1 * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, in_channels, output_length), dtype=in_3.dtype, device=in_3.device)
    
    # Set up Triton kernel launch (3D grid: batch, channel, seq_offset)
    BLOCK_SIZE = 128  # Number of sequence positions processed per program
    
    # Grid dimensions
    batch_dim = batch_size
    channel_dim = in_channels
    seq_dim = (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_avg_pool1d_kernel[(batch_dim, channel_dim, seq_dim)](
        input_ptr=in_3,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_avg_pool1d