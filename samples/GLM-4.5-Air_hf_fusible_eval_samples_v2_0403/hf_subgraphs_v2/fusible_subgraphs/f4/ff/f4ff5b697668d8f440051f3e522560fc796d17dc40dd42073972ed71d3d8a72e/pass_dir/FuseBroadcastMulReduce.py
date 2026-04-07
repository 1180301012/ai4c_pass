import torch
import triton
import triton.language as tl

def pattern(softmax_input, in_0, in_1):
    """
    Pattern matching for the optimized computation:
    - softmax_input: result of torch.nn.functional.softmax(input, dim=2)
    - in_0: [1,1,1,64] tensor for x-direction computation
    - in_1: [1,1,64,1] tensor for y-direction computation
    """
    reshaped = softmax_input.reshape(-1, 17, 64, 64)
    
    # First computation path (x-direction)
    mul_x = reshaped.mul(in_0)
    reshaped_x = mul_x.reshape(-1, 17, -1)
    sum_x = torch.sum(reshaped_x, dim=2, keepdim=True)
    
    # Second computation path (y-direction)  
    mul_y = reshaped.mul(in_1)
    reshaped_y = mul_y.reshape(-1, 17, -1)
    sum_y = torch.sum(reshaped_y, dim=2, keepdim=True)
    
    # Combine results
    final_result = torch.cat([sum_x, sum_y], dim=-1)
    
    return reshaped, final_result

def replacement_args(softmax_input, in_0, in_1):
    return (softmax_input, in_0, in_1)

@triton.jit
def fused_broadcast_mul_reduce_kernel(
    softmax_ptr,
    in_0_ptr,
    in_1_ptr,
    sum_x_ptr,
    sum_y_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    SPATIAL_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that fuses:
    - Broadcasting multiplication operations  
    - Sum reduction operations
    
    Computes both x and y directions simultaneously.
    """
    # Calculate program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges within the block
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundary conditions
    m_mask = m_offsets < N
    n_mask = n_offsets < H
    
    # Load broadcast tensors once per program
    if pid_m == 0 and pid_n == 0:  # Only load once to reduce memory traffic
        # in_0: [1,1,1,64] → reshape to [64] 
        in_0_data = tl.load(in_0_ptr, mask=tl.arange(0, 64) < 64, other=0.0)
        # in_1: [1,1,64,1] → reshape to [64]
        in_1_data = tl.load(in_1_ptr, mask=tl.arange(0, 64) < 64, other=0.0)
    
    # Each program processes one (m,n) pair
    if m_mask and n_mask:
        # Calculate linear index for this (m,n) pair
        linear_idx = m_offsets * H + n_offsets
        
        # Load softmax data for this (m,n) pair: (4096,) 
        softmax_base_idx = linear_idx * SPATIAL_SIZE
        softmax_data = tl.load(softmax_ptr + softmax_base_idx + tl.arange(0, SPATIAL_SIZE),
                              mask=tl.arange(0, SPATIAL_SIZE) < SPATIAL_SIZE,
                              other=0.0)
        
        # Reshape to (64, 64) for spatial processing
        softmax_2d = softmax_data.reshape(64, 64)
        
        # Broadcast multiply for x-direction: simulate in_0 [1,1,1,64] broadcasting
        # This multiplies each row (first dim) by the in_0 values
        mul_x = softmax_2d * in_0_data[None, :]  # (64,64) * (1,64) → (64,64)
        
        # Broadcast multiply for y-direction: simulate in_1 [1,1,64,1] broadcasting  
        # This multiplies each column (second dim) by the in_1 values
        mul_y = softmax_2d * in_1_data[:, None]  # (64,64) * (64,1) → (64,64)
        
        # Sum reduction over all 4096 spatial elements
        sum_x_result = tl.sum(mul_x)
        sum_y_result = tl.sum(mul_y)
        
        # Store results
        tl.store(sum_x_ptr + linear_idx, sum_x_result)
        tl.store(sum_y_ptr + linear_idx, sum_y_result)

@torch.fx.wrap
def fused_broadcast_mul_reduce(softmax_input, in_0, in_1):
    """
    Wrapper function to launch the optimized kernel
    """
    # Get tensor shapes and sizes
    N = softmax_input.shape[0]  # First dimension (varies: 1, 4, 64, 128, 256, 512)
    H = 17
    spatial_size = 64 * 64  # 4096 spatial elements per (N, H) pair
    
    # Create output tensors
    sum_x = torch.empty((N, H), dtype=softmax_input.dtype, device=softmax_input.device)
    sum_y = torch.empty((N, H), dtype=softmax_input.dtype, device=softmax_input.device)
    
    # Tile sizes for optimal GPU utilization
    BLOCK_SIZE_M = 32  # Process multiple N elements per block
    BLOCK_SIZE_N = 8   # Process multiple H dimensions per block
    
    # Calculate grid size - each thread block processes a block of (N, H) pairs
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_broadcast_mul_reduce_kernel[grid_m, grid_n](
        softmax_input,
        in_0,
        in_1,
        sum_x,
        sum_y,
        N,
        H,
        spatial_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    # Apply the original logic: reshape and concatenate
    reshaped_input = softmax_input.reshape(N, H, 64, 64)
    sum_x_unsqueeze = sum_x.unsqueeze(-1)  # (N, H, 1, 1)
    sum_y_unsqueeze = sum_y.unsqueeze(-1)  # (N, H, 1, 1)
    final_result = torch.cat([sum_x_unsqueeze, sum_y_unsqueeze], dim=-1)  # (N, H, 1, 2)
    
    return reshaped_input, final_result

def replacement_func():
    return fused_broadcast_mul_reduce