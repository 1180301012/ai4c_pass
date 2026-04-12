import torch
import triton
import triton.language as tl

def pattern(in_4, in_1):
    """Match just the einsum operation"""
    result = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return result

def replacement_args(in_4, in_1):
    """Extract arguments needed for the replacement"""
    return (in_4, in_1)

@triton.jit
def optimized_einsum_kernel_v2(
    value_4_ptr,           # [B, C, H, W] - input tensor
    getitem_3_ptr,         # [B, H, W, J] - input tensor  
    out_ptr,               # [B, C, H, W] - output tensor
    output_dtype: tl.constexpr,
    B, C, H, W, J,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized kernel with improved block sizes"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Bounds checking
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offset < B * C
    n_mask = n_offset < H * W
    
    # Convert to 2D offsets
    m_offset_2d = m_offset // C  # batch dimension
    c_offset = m_offset % C       # channel dimension
    
    n_offset_2d = n_offset // W  # height dimension  
    w_offset = n_offset % W      # width dimension
    
    # Matrix multiplication loop
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, J, BLOCK_SIZE_K):
        k_mask = k < J
        
        # Load value_4: [B, C, H, W] -> reshape for matmul
        # We want [B*C, H*W] matrix
        offsets_4 = (m_offset_2d * H * W + n_offset_2d) * J + k
        offsets_4_mask = m_mask[:, None] & n_mask[None, :] & k_mask
        value_4 = tl.load(value_4_ptr + offsets_4, mask=offsets_4_mask, other=0.0)
        value_4 = value_4.to(tl.float32)
        
        # Load getitem_3: [B, H, W, J] -> reshape for matmul  
        # We want [B*H*W, J] matrix
        m_3_offset = (m_offset_2d * H + n_offset_2d) * J + k
        offsets_3_mask = m_mask[:, None] & n_mask[None, :] & k_mask
        getitem_3 = tl.load(getitem_3_ptr + m_3_offset, mask=offsets_3_mask, other=0.0)
        getitem_3 = getitem_3.to(tl.float32)
        
        # Accumulate - use simple multiplication and sum reduction
        acc += tl.sum(value_4 * getitem_3, axis=0)
    
    # Store result
    out_offsets = (m_offset[:, None] * H * W + n_offset[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    
    # Convert to output dtype
    if output_dtype == 0:  # float16/bfloat16
        result = acc.to(tl.float16)
    else:  # float32
        result = acc.to(tl.float32)
    tl.store(out_ptr + out_offsets, result, mask=out_mask)

@torch.fx.wrap
def optimized_einsum_call_v2(in_4, in_1):
    """Call the optimized einsum kernel with better block sizes"""
    B, C, H, W = in_4.shape
    J = in_1.shape[-1]
    
    # More sophisticated block size selection based on tensor dimensions
    total_elements = B * C * H * W
    if total_elements > 500000:
        # Large tensors: use larger blocks
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    elif total_elements > 100000:
        # Medium tensors: use medium blocks
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 8
    else:
        # Small tensors: use smaller blocks to reduce overhead
        BLOCK_SIZE_M = 8
        BLOCK_SIZE_N = 8
        BLOCK_SIZE_K = 4
    
    # Calculate grid size
    M = (B * C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    N = (H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty((B, C, H, W), dtype=in_4.dtype, device=in_4.device)
    
    # Determine dtype parameter
    output_dtype = 0 if in_4.element_size() == 2 else 1
    
    # Launch kernel
    optimized_einsum_kernel_v2[(M, N)](
        value_4_ptr=in_4,
        getitem_3_ptr=in_1,
        out_ptr=out,
        output_dtype=output_dtype,
        B=B, C=C, H=H, W=W, J=J,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    """Return the optimized einsum function"""
    return optimized_einsum_call_v2