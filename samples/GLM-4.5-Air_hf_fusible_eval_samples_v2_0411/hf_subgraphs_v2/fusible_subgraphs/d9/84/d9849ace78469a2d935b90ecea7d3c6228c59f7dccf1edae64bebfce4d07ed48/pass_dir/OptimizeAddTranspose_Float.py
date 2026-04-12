import torch
import triton
import triton.language as tl

# Pattern matching function - match the entire computation: add + transpose
def pattern(x, y):
    """Match: broadcast add + transpose sequence"""
    # Match: in_1 += in_0  (equivalent to in_1 = in_1 + in_0)
    # We use regular addition but the semantics match the in-place operation
    added = x + y
    # Match: in_2 = in_1 (assignment)
    # Match: tmp_2 = in_2.transpose(1, 2) 
    result = added.transpose(1, 2)
    return result

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized Triton kernel for fused add + transpose
@triton.jit
def fused_add_transpose_kernel(
    x_ptr,           # [128, 1] tensor
    y_ptr,           # [1, 128, 19] tensor  
    out_ptr,         # [1, 19, 128] output tensor
    n_m: tl.constexpr,  # num_features = 128
    n_t: tl.constexpr,  # num_time = 19
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    # Load coordinates
    m_offset = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    t_offset = tl.program_id(1) * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    
    # Create 2D grid
    m_mask = m_offset < n_m
    t_mask = t_offset < n_t
    
    # Broadcast x from [128, 1] to [128, 19] and load
    x = tl.load(x_ptr + m_offset, mask=m_mask, other=0.0)
    x = x.to(tl.float16)  # Ensure float16 for consistency
    
    # Load y from [1, 128, 19] - we need to reshape to 2D for efficient access
    # Reshape y from [1, 128, 19] to [128, 19] 
    y = tl.load(y_ptr + (m_offset[:, None] * n_t + t_offset[None, :]).to(tl.int64), 
               mask=m_mask[:, None] & t_mask[None, :], other=0.0)
    y = y.to(tl.float16)  # Ensure float16 for consistency
    
    # Add x broadcasted to [128, 19] to y
    # x is [128, 1], we need to broadcast to [128, 19]
    x_bcast = x[:, None]  # Reshape x to [128, 1]
    out = x_bcast + y
    
    # Transpose result from [128, 19] to [19, 128] and store
    # We need to access as [19, 128] dimension order
    out_transposed = tl.store(out_ptr + (t_offset[:, None] * n_m + m_offset[None, :]).to(tl.int64),
                            out,
                            mask=t_mask[:, None] & m_mask[None, :])

# Kernel wrapper
@torch.fx.wrap  
def fused_add_transpose(x, y):
    """Fused addition and transpose operation"""
    n_m = x.shape[0]  # 128
    n_t = y.shape[2]  # 19
    
    # Calculate optimal block sizes
    BLOCK_SIZE_M = 64  # Process multiple M elements per thread
    BLOCK_SIZE_T = 32  # Process multiple T elements per thread
    
    # Calculate grid dimensions  
    num_blocks_m = (n_m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_t = (n_t + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    
    # Create output tensor with transposed shape [1, 19, 128]
    output_shape = (1, n_t, n_m)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_add_transpose_kernel[(num_blocks_m, num_blocks_t)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_m=n_m,
        n_t=n_t,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_add_transpose