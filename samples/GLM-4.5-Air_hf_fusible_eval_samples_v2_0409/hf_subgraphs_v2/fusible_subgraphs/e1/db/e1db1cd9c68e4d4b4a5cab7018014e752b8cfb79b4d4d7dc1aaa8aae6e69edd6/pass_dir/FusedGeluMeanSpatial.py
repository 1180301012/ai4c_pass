import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_mean_kernel(
    input_ptr,
    gelu_out_ptr,
    mean_out_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program identifier
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory offsets for batch and channel
    batch_offset = pid_m * n_channels * height * width
    channel_offset = pid_n * height * width
    
    # Initialize shared memory for partial sums
    shared_sums = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    shared_sums = tl.core.tensor.SharedArray(BLOCK_M, BLOCK_N, dtype=tl.float32, evict=True)
    
    # Calculate per-thread spatial range
    spatial_size = height * width
    elements_per_thread = (spatial_size + BLOCK_M * BLOCK_N - 1) // (BLOCK_M * BLOCK_N)
    thread_spatial_start = tl.program_id(2) * elements_per_thread
    thread_spatial_end = min(thread_spatial_start + elements_per_thread, spatial_size)
    
    # Initialize thread sums
    thread_sum = 0.0
    
    # Process spatial elements assigned to this thread
    for spatial_idx in range(thread_spatial_start, thread_spatial_end):
        h = spatial_idx // width
        w = spatial_idx % width
        
        # Global memory offset
        global_offset = batch_offset + channel_offset + h * width + w
        
        # Load input
        x = tl.load(input_ptr + global_offset, mask=spatial_idx < spatial_size, other=0.0)
        
        # Compute GELU (approximation for performance)
        gelu_val = x * 0.5 * (1.0 + tl.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
        
        # Store GELU output
        tl.store(gelu_out_ptr + global_offset, gelu_val, mask=spatial_idx < spatial_size)
        
        # Accumulate for mean computation
        thread_sum += gelu_val.to(tl.float32)
    
    # Store thread partial sum in shared memory
    m_idx = tl.program_id(2) // BLOCK_N
    n_idx = tl.program_id(2) % BLOCK_N
    if m_idx < BLOCK_M and n_idx < BLOCK_N:
        tl.store(shared_sums + m_idx * BLOCK_N + n_idx, thread_sum)
    
    # Barrier for shared memory synchronization
    tl.sync()
    
    # Perform parallel reduction in shared memory
    if m_idx < BLOCK_M and n_idx < BLOCK_N:
        # Reduce within thread block
        stride = 1
        while stride < BLOCK_M * BLOCK_N:
            idx = m_idx * BLOCK_N + n_idx
            if idx + stride < BLOCK_M * BLOCK_N:
                other_sum = tl.load(shared_sums + idx + stride)
                new_sum = tl.load(shared_sums + idx) + other_sum
                tl.store(shared_sums + idx, new_sum)
            stride *= 2
            tl.sync()
        
        # Store final mean for this (batch, channel) pair
        if n_idx == 0 and m_idx == 0:
            final_sum = tl.load(shared_sums)
            mean_val = final_sum / (height * width)
            
            # Store mean output at [pid_m, pid_n, 0, 0]
            mean_offset = pid_m * n_channels + pid_n
            tl.store(mean_out_ptr + mean_offset * 4, mean_val)  # 4 bytes for float32

@torch.fx.wrap
def fused_gelu_mean(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    
    # Output tensors
    gelu_out = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    mean_out = torch.empty((batch_size, channels, 1, 1), dtype=torch.float32, device=input_tensor.device)
    
    # Calculate grid dimensions
    n_blocks_m = (batch_size + 31) // 32  # Block size for dimension 0
    n_blocks_n = (channels + 31) // 32    # Block size for dimension 1
    n_blocks_spatial = ((height * width) + 1023) // 1024  # Spatial blocks
    
    # Launch kernel
    fused_gelu_mean_kernel[(n_blocks_m * n_blocks_n * n_blocks_spatial,)](
        input_tensor,
        gelu_out,
        mean_out.view(-1),  # Flatten spatial dimensions for easier indexing
        batch_size,
        channels,
        height,
        width,
        32,  # BLOCK_M (threads per block for batch)
        32,  # BLOCK_N (threads per block for channels)
    )
    
    return gelu_out, mean_out

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def fused_wrapper(in_0):
        return fused_gelu_mean(in_0)
    return fused_wrapper