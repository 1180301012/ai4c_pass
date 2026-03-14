import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    # Match the exact computation from the model
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton - simplified version without autotune for now
@triton.jit
def optimized_fused_mean_kernel(
    in_ptr,
    out_ptr,
    n_batch,
    n_height,
    n_channels,
    n_width1,
    n_width2,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
@triton.jit
def optimized_fused_mean_kernel(
    in_ptr,
    out_ptr,
    n_batch,
    n_height,
    n_channels,
    n_width1,
    n_width2,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """High-performance fused kernel with autotuning"""
    program_id = tl.program_id(0)
    
    batch_idx = program_id // n_height
    height_idx = program_id % n_height
    
    if batch_idx >= n_batch or height_idx >= n_height:
        return
    
    # Shared memory for partial sums (warps can collaborate)
    warp_sum = tl.zeros(32, dtype=tl.float32)  # 32 threads per warp
    
    # Each thread computes the mean for one slice of work
    channel_start = tl.program_id(1) * (n_channels + 31) // 32
    channel_end = min((tl.program_id(1) + 1) * (n_channels + 31) // 32, n_channels)
    
    local_sum = 0.0
    local_count = 0
    
    # Main computation loop - optimized memory access pattern
    for c in range(channel_start, channel_end):
        for w1 in range(0, n_width1, BLOCK_SIZE // 32):
            for w2 in range(0, n_width2):
                for w1_inner in range(w1, min(w1 + BLOCK_SIZE // 32, n_width1)):
                    # Memory coalescing pattern
                    offset = (batch_idx * n_channels * n_height * n_width1 * n_width2 +
                             c * n_height * n_width1 * n_width2 +
                             height_idx * n_width1 * n_width2 +
                             w1_inner * n_width2 + w2)
                    
                    val = tl.load(in_ptr + offset, other=0.0)
                    local_sum += val
                    local_count += 1
    
    # Warp reduction
    warp_id = tl.program_id(1) % 32
    tl.device_assert(warp_id < 32, "Invalid warp ID")
    
    # Add to shared memory
    if warp_id < local_count:
        warp_sum[warp_id] = local_sum
    
    # Synchronize within the warp
    tl.device_barrier()
    
    # Global reduction warps
    global_sum = 0.0
    for i in range(32):
        global_sum += warp_sum[i]
    
    total_elements = n_channels * n_width1 * n_width2
    if total_elements > 0:
        mean_value = global_sum / total_elements
    else:
        mean_value = 0.0
    
    # Write result
    result_offset = batch_idx * n_height + height_idx
    tl.store(out_ptr + result_offset, mean_value)

# Highly optimized wrapper with Triton kernel
@torch.fx.wrap
def optimized_fused_mean(in_0):
    """Optimized fused computation using high-performance Triton kernel"""
    batch_size, channels, height, width1, width2 = in_0.shape
    
    # Output shape: [batch_size, height, 1, 1]
    output = torch.empty((batch_size, height), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with autotuning
    n_programs = batch_size * height
    
    optimized_fused_mean_kernel[(n_programs, 32)](
        in_ptr=in_0,
        out_ptr=output,
        n_batch=batch_size,
        n_height=height,
        n_channels=channels,
        n_width1=width1,
        n_width2=width2,
        BLOCK_SIZE=256,
    )
    
    # Reshape to final output shape
    return output.view(batch_size, height, 1, 1)

# Replacement function
def replacement_func():
    return optimized_fused_mean