import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: x.mean((2, 3))
def pattern(x):
    return x.mean((2, 3))

# Argument extraction function
# Extracts input tensor 'x'
def replacement_args(x):
    return (x,)

# Optimized kernel for mean reduction over (2,3)
@triton.jit
def mean_reduction_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduces over spatial dimensions (H, W) using shared memory reduction."""
    # Each block handles one channel
    c_id = tl.program_id(0)
    
    # Each thread processes one spatial element for this channel
    spatial_index = tl.thread_id(0)
    spatial_size = height * width
    
    # Skip if thread index exceeds spatial_size
    if spatial_index >= spatial_size:
        return

    # Calculate global index in input tensor (B, C, H, W)
    # For channel c_id, the element at spatial_index is at:
    #   spatial_index * channels + c_id
    idx = spatial_index * channels + c_id
    val = tl.load(x_ptr + idx)

    # Store value in shared memory
    shmem = tl.shared_memory(BLOCK_SIZE, dtype=tl.float32)
    shmem[spatial_index] = val
    
    # Synchronize threads for shared memory
    tl.sync_threads()

    # Reduction in shared memory
    block_size = BLOCK_SIZE
    while block_size > 1:
        if spatial_index < block_size // 2:
            shmem[spatial_index] += shmem[spatial_index + block_size // 2]
        block_size //= 2
        tl.sync_threads()

    # Thread 0 stores the final result
    if spatial_index == 0:
        tl.store(out_ptr + c_id, shmem[0])

# Kernel wrapper
@torch.fx.wrap
def mean_reduction(x):
    B, C, H, W = x.shape
    spatial_size = H * W
    
    # Allocate output tensor (B, C)
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    
    # Block size must be power of two and >= spatial_size
    BLOCK_SIZE = 256
    
    # Launch kernel with one block per channel
    mean_reduction_kernel[(C, 1)](
        x_ptr=x,
        out_ptr=out,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Divide by spatial_size to get mean
    out = out / spatial_size
    return out

# Replacement function - returns the optimized kernel
# Note: Must be a zero-argument function
def replacement_func():
    return mean_reduction