import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


@triton.jit
def transpose_kernel(
    x_ptr,  # in_2: [1, 16, 196, 48]
    out_ptr,  # output: [1, 16, 48, 196]
    batch_size,
    channels,
    dim0,
    dim1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one output (batch, channels)
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Calculate starting positions for this program
    x_base = x_ptr + (batch_idx * channels + channel_idx) * dim0 * dim1 * 2
    out_base = out_ptr + (batch_idx * channels + channel_idx) * dim1 * dim0 * 2
    
    # Process blocks
    for i in range(0, dim0, BLOCK_M):
        for j in range(0, dim1, BLOCK_N):
            # Load x values [dim0, dim1] -> [BLOCK_M, BLOCK_N]
            x_offsets = (tl.arange(0, BLOCK_M)[:, None] * dim1 + tl.arange(0, BLOCK_N)[None, :]) * 2
            x_mask = (tl.arange(0, BLOCK_M)[:, None] + i) < dim0
            x_mask &= (tl.arange(0, BLOCK_N)[None, :] + j) < dim1
            x = tl.load(x_base + x_offsets, mask=x_mask, other=0.0)
            
            # Transpose to [BLOCK_N, BLOCK_M] and store
            out_offsets = (tl.arange(0, BLOCK_N)[:, None] * dim0 + tl.arange(0, BLOCK_M)[None, :]) * 2
            out_mask = (tl.arange(0, BLOCK_N)[:, None] + j) < dim1
            out_mask &= (tl.arange(0, BLOCK_M)[None, :] + i) < dim0
            tl.store(out_base + out_offsets, x, mask=out_mask)


@torch.fx.wrap
def optimized_transpose(in_2):
    # Input shape: in_2 [1, 16, 196, 48]
    batch_size = 1
    channels = 16
    dim0 = 196
    dim1 = 48
    
    # Prepare output tensor
    output_shape = [batch_size, channels, dim1, dim0]
    out = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Set up triton kernel parameters
    BLOCK_M = 32  # First dimension to transpose
    BLOCK_N = 32  # Second dimension to transpose
    
    # Calculate grid size
    num_programs = batch_size * channels
    
    # Launch kernel
    transpose_kernel[(num_programs,)](
        in_2, out,
        batch_size, channels, dim0, dim1,
        BLOCK_M, BLOCK_N
    )
    
    return out


def replacement_func():
    return optimized_transpose