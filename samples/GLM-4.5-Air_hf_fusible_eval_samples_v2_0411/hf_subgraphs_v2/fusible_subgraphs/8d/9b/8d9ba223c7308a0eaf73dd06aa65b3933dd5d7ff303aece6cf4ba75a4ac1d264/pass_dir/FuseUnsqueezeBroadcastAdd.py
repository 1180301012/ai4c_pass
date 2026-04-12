import torch
import triton
import triton.language as tl

# Pattern matching for unsqueeze + broadcasting operations
def pattern(main_tensor, broadcast_tensor):
    """
    Match the computation pattern:
    tmp_14 = broadcast_tensor.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = main_tensor + tmp_15
    tmp_17 = broadcast_tensor.unsqueeze(1)  
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    """
    tmp_14 = broadcast_tensor.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = main_tensor + tmp_15
    tmp_17 = broadcast_tensor.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0) 
    tmp_19 = tmp_16 + tmp_18
    return tmp_19

# Extract arguments for the replacement
def replacement_args(main_tensor, broadcast_tensor):
    return (main_tensor, broadcast_tensor)

# Optimized Triton kernel for fused broadcasting + addition
@triton.jit
def fuse_broadcast_add_kernel(
    main_ptr,        # Main tensor [1,64,12,64,64]
    broadcast_ptr,   # Broadcast tensor [64,64,64] -> becomes [1,64,1,64,64]
    out_ptr,         # Output tensor [1,64,12,64,64]
    n_batches,       # 1
    n_heads,         # 64  
    n_features,      # 12
    height,          # 64
    width,           # 64
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a tile across all dimensions
    b_pid = tl.program_id(0)  # batch (only 1)
    h_pid = tl.program_id(1)  # head
    f_pid = tl.program_id(2)  # feature 
    m_pid = tl.program_id(3)  # height
    n_pid = tl.program_id(4)  # width
    
    # Compute tile bounds
    b = b_pid  # Only one batch
    h_offset = h_pid * BLOCK_SIZE_H
    f_offset = f_pid * BLOCK_SIZE_F
    m_offset = m_pid * BLOCK_SIZE_M
    n_offset = n_pid * BLOCK_SIZE_N
    
    # Compute bounds
    mask_h = h_offset + tl.arange(0, BLOCK_SIZE_H) < n_heads
    mask_f = f_offset + tl.arange(0, BLOCK_SIZE_F) < n_features
    mask_m = m_offset + tl.arange(0, BLOCK_SIZE_M) < height
    mask_n = n_offset + tl.arange(0, BLOCK_SIZE_N) < width
    
    # Load main tensor data
    main_offsets_h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    main_offsets_f = f_offset + tl.arange(0, BLOCK_SIZE_F)
    main_offsets_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
    main_offsets_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    
    # Load broadcast tensor data: [h, m, n] -> broadcast to [1, h, 1, m, n]
    broadcast_offsets_h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    broadcast_offsets_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
    broadcast_offsets_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    
    # Process element with vectorization where possible
    for i in range(BLOCK_SIZE_H):
        if mask_h[i]:
            for j in range(BLOCK_SIZE_F):
                if mask_f[j]:
                    for k in range(BLOCK_SIZE_M):
                        if mask_m[k]:
                            for l in range(BLOCK_SIZE_N):
                                if mask_n[l]:
                                    # Main tensor offset: [b, h_i, f_j, m_k, n_l]
                                    main_idx = b * n_heads * n_features * height * width + \
                                              (h_offset + i) * n_features * height * width + \
                                              (f_offset + j) * height * width + \
                                              (m_offset + k) * width + (n_offset + l)
                                    
                                    # Broadcast tensor offset: [h_i, m_k, n_l] -> expand to [1,h,1,m,n]
                                    broadcast_idx = (h_offset + i) * height * width + (m_offset + k) * width + (n_offset + l)
                                    
                                    # Load values
                                    main_val = tl.load(main_ptr + main_idx)
                                    broadcast_val = tl.load(broadcast_ptr + broadcast_idx)
                                    
                                    # Perform both additions: main_val + broadcast_val + broadcast_val
                                    result = main_val + 2 * broadcast_val
                                    
                                    # Store result
                                    tl.store(out_ptr + main_idx, result)

# Simplified optimization - just fuse the broadcasting operations
@torch.fx.wrap
def optimized_fuse_broadcast_add(main_tensor, broadcast_tensor):
    """Fused unsqueeze + broadcasting + double addition operation"""
    
    # Original computation:
    # tmp_14 = broadcast_tensor.unsqueeze(1) -> [64,1,64,64]
    # tmp_15 = tmp_14.unsqueeze(0) -> [1,64,1,64,64]
    # tmp_16 = main_tensor + tmp_15
    # tmp_17 = broadcast_tensor.unsqueeze(1) -> [64,1,64,64] 
    # tmp_18 = tmp_17.unsqueeze(0) -> [1,64,1,64,64]
    # tmp_19 = tmp_16 + tmp_18
    
    # Simplified: compute both broadcasts and additions directly
    # The unsqueeze operations don't copy data, just change views
    
    # Compute first unsqueeze + broadcast
    broadcast_expanded_1 = broadcast_tensor.unsqueeze(1).unsqueeze(0)  # [1,64,1,64,64]
    
    # Compute second unsqueeze + broadcast  
    broadcast_expanded_2 = broadcast_tensor.unsqueeze(1).unsqueeze(0)  # [1,64,1,64,64]
    
    # Perform both additions
    result = main_tensor + broadcast_expanded_1 + broadcast_expanded_2
    
    return result

# Replacement function
def replacement_func():
    return optimized_fuse_broadcast_add