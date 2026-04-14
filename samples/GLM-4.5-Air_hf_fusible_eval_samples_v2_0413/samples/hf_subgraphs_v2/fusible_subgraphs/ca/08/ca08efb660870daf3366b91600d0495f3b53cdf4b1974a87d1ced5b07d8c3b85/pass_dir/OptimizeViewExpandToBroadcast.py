import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """Match the computation pattern: view(shape=[1, 2, 1, 8, 8]) followed by expand(shape=[1, 2, 64, 8, 8])"""
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Triton kernel for optimized broadcasting to [1, 2, 64, 8, 8]
@triton.jit
def broadcast_to_kernel(
    in_ptr,
    out_ptr,
    n_batch,
    n_channels,
    n_h,
    n_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that broadcasts input from shape [1, 2, 8, 8] to [1, 2, 64, 8, 8]"""
    # Create a 2D grid (batch, channels)
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    batch = pid_bc // n_channels
    channel = pid_bc % n_channels
    
    h_offset = pid_h * BLOCK_SIZE
    w_offset = pid_w * BLOCK_SIZE
    
    # Load and process input block (original shape [1, 2, 8, 8])
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE)
    w_indices = w_offset + tl.arange(0, BLOCK_SIZE)
    masks_h = h_indices < n_h
    masks_w = w_indices < n_w
    
    in_block = tl.load(
        in_ptr + batch * (n_channels * n_h * n_w) + 
        channel * (n_h * n_w) + 
        h_indices[:, None] * n_w + w_indices[None, :],
        mask=masks_h[:, None] & masks_w[None, :],
        other=0.0,
    )
    
    # Store the block repeated for all 64 expansions along the new dimension
    for exp_dim in range(64):  # Broadcast across 64 positions
        out_offset = batch * (n_channels * 64 * n_h * n_w) + \
                    channel * (64 * n_h * n_w) + \
                    exp_dim * (n_h * n_w)
        
        tl.store(
            out_ptr + out_offset + h_indices[:, None] * n_w + w_indices[None, :],
            in_block,
            mask=masks_h[:, None] & masks_w[None, :],
        )


@triton.jit
def optimized_broadcast_kernel(
    in_ptr, 
    out_ptr,
    n_batch,
    n_channels, 
    n_h,
    n_w
):
    """Highly optimized broadcast kernel for [1, 2, 8, 8] -> [1, 2, 64, 8, 8]"""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1) 
    pid_h = tl.program_id(2)
    
    # For our workload (small tensor), each thread handles one (batch, channel, h) position
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    h = tl.program_id(2)
    
    # Broadcast all 64 expansion dimensions for this (batch, channel, h)
    for exp_k in range(64):
        # Calculate source offset (original shape [1, 2, 8, 8])
        src_offset = (batch * (n_channels * n_h * n_w) + 
                     channel * (n_h * n_w) + 
                     h * n_w)
        
        # Calculate all target offsets for this expansion dimension
        for w in range(n_w):
            # Target offset with expansion (target shape [1, 2, 64, n_h, n_w])
            tgt_offset = (batch * (n_channels * 64 * n_h * n_w) + 
                         channel * (64 * n_h * n_w) + 
                         exp_k * (n_h * n_w) + 
                         h * n_w + w)
            
            # Copy data
            src_val = tl.load(in_ptr + src_offset + w)
            tl.store(out_ptr + tgt_offset, src_val)


@torch.fx.wrap
def optimized_broadcast_triton(in_0):
    """High-performance broadcasting using efficient Triton memory operations"""
    n_batch, n_channels, n_h, n_w = in_0.shape[0], in_0.shape[1], in_0.shape[2], in_0.shape[3]
    
    # Create output tensor with target shape [1, 2, 64, n_h, n_w]
    tmp_3 = torch.empty((n_batch, n_channels, 64, n_h, n_w), device=in_0.device, dtype=in_0.dtype)
    
    # Use efficient single-threaded copy for this small tensor
    # We'll copy each expansion dimension using Triton operations
    BLOCK_SIZE = 8
    
    # Create 3D grid for parallel broadcasting
    grid_n_batch = (n_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n_channels = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE  
    grid_n_h = (n_h + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_broadcast_kernel[(grid_n_batch, grid_n_channels, grid_n_h)](
        in_0, tmp_3, n_batch, n_channels, n_h, n_w
    )
    
    return tmp_3


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_broadcast_triton