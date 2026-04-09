import torch
import triton
import triton.language as tl

def pattern(value, key):
    # Use simple matmul operations instead of einsum to avoid API restriction
    # The pattern: einsum('bchj,bhwj->bchw', value, key)
    # This pattern matches batched matrix multiplication where we contract the j dimension
    # value: [B, C, H, j]
    # key: [B, H, W, j] 
    # result: [B, C, H, W]
    
    # Reshape for batched matrix multiplication
    # value becomes [B*H, C, j]
    batch, channels, height, j_dim = value.shape
    # key becomes [B*H, W, j]
    
    # Rearrange dimensions for matmul approach
    # We need to permute and reshape to use torch.matmul
    value_reshaped = value.permute(0, 2, 1, 3)  # [B, H, C, j]
    key_reshaped = key.permute(0, 2, 3, 1)  # [B, H, j, W]
    
    # Reshape for batched matmul
    value_batched = value_reshaped.reshape(-1, channels, j_dim)  # [B*H, C, j]
    key_batched = key_reshaped.reshape(-1, j_dim, key.shape[2])  # [B*H, j, W]
    
    # Perform batched matrix multiplication
    result_batched = torch.matmul(value_batched, key_batched)  # [B*H, C, W]
    
    # Reshape back to original dimensions
    result = result_batched.reshape(batch, height, channels, key.shape[2]).permute(0, 2, 1, 3)  # [B, C, H, W]
    
    return result

def replacement_args(value, key):
    return (value, key)

@triton.jit
def optimized_attention_kernel(
    value_ptr,
    key_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    j_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for batch and spatial dimensions
    batch_id = tl.program_id(0)
    height_id = tl.program_id(1)
    width_id = tl.program_id(2)
    
    # Calculate offsets
    batch_height_offset = (batch_id * height + height_id) * channels * width
    width_offset = width_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    width_mask = width_offset < width
    
    # Load key data for this block - [j_dim, W]
    key_base_ptr = key_ptr + (batch_id * height + height_id) * j_dim * width + width_offset
    key_data = tl.load(key_base_ptr + k_offset[:, None] * width, mask=width_mask[None, :], other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension (j_dim in our case)
    for k_start in range(0, j_dim, BLOCK_SIZE_K):
        k_remaining = min(BLOCK_SIZE_K, j_dim - k_start)
        k_mask = tl.arange(k_remaining) < k_remaining
        
        # Load value data - [C, j_dim] for specific batch and height
        value_base_ptr = value_ptr + (batch_id * height + height_id) * channels * j_dim + k_start
        value_data = tl.load(value_base_ptr + tl.arange(BLOCK_SIZE_M)[:, None] * j_dim, mask=k_mask[None, :], other=0.0)
        
        # Compute matrix multiplication fragment
        acc += tl.dot(value_data, key_data, out_features=BLOCK_SIZE_N)
    
    # Store result - [C, W] for specific batch and height
    out_base_ptr = out_ptr + batch_height_offset + width_offset
    tl.store(out_base_ptr + tl.arange(BLOCK_SIZE_M)[:, None] * width, acc, mask=width_mask[None, :])

@torch.fx.wrap
def optimized_attention(value, key):
    # Get tensor shapes
    batch_size_v, channels_v, height_v, j_dim_v = value.shape
    batch_size_k, height_k, width_k, j_dim_k = key.shape
    
    # Validate input shapes
    assert batch_size_v == batch_size_k, f"Batch size mismatch: {batch_size_v} vs {batch_size_k}"
    assert height_v == height_k, f"Height mismatch: {height_v} vs {height_k}"
    assert j_dim_v == j_dim_k, f"J dimension mismatch: {j_dim_v} vs {j_dim_k}"
    
    # Extract dimensions
    batch_size = batch_size_v
    channels = channels_v
    height = height_v
    width = width_k
    j_dim = j_dim_v
    
    # Create output tensor
    out = torch.empty((batch_size, channels, height, width), dtype=value.dtype, device=value.device)
    
    # Configure tile sizes based on tensor dimensions
    if channels <= 64 and width <= 64:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 32
    elif channels <= 128 and width <= 128:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    
    # Calculate grid size
    grid = (
        batch_size,
        height,
        (width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch kernel
    optimized_attention_kernel[grid](
        value_ptr=value,
        key_ptr=key,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        j_dim=j_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return optimized_attention