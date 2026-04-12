import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match the entire computation sequence"""
    tmp_1 = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    tmp_2 = in_3 + tmp_1  # Regular addition instead of in-place to match pattern expectation
    tmp_3 = tmp_2 * in_0
    tmp_4 = tmp_3 + in_2
    return tmp_4.contiguous()

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract all arguments needed for the replacement"""
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def einsum_opt_kernel(
    value_4_ptr,           # [B, C, H, W] - input tensor
    getitem_3_ptr,         # [B, H, W, J] - input tensor  
    out_ptr,               # [B, C, H, W] - output tensor
    B, C, H, W, J,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized kernel for einsum 'bchj,bhwj->bchw'"""
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
        
        # Accumulate
        acc += tl.dot(value_4, getitem_3)
    
    # Store result
    out_offsets = (m_offset[:, None] * H * W + n_offset[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    
    # Convert to output dtype
    result = acc.to(tl.float16 if out_ptr.dtype.element_size() == 2 else tl.float32)
    tl.store(out_ptr + out_offsets, result, mask=out_mask)

@triton.jit
def fused_kernel(
    value_4_ptr,           # [B, C, H, W] - input tensor
    getitem_3_ptr,         # [B, H, W, J] - input tensor  
    out_333_ptr,           # [B, C, H, W] - input tensor for addition
    gamma_params_ptr,      # scalar parameter
    out_ptr,               # [B, C, H, W] - final output
    B, C, H, W, J,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel: einsum + add + multiply + add"""
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
    
    # Load scalar gamma parameter
    gamma = tl.load(gamma_params_ptr)
    
    # Matrix multiplication loop
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, J, BLOCK_SIZE_K):
        k_mask = k < J
        
        # Load value_4: [B, C, H, W]
        offsets_4 = (m_offset_2d * H * W + n_offset_2d) * J + k
        offsets_4_mask = m_mask[:, None] & n_mask[None, :] & k_mask
        value_4 = tl.load(value_4_ptr + offsets_4, mask=offsets_4_mask, other=0.0)
        value_4 = value_4.to(tl.float32)
        
        # Load getitem_3: [B, H, W, J]
        m_3_offset = (m_offset_2d * H + n_offset_2d) * J + k
        offsets_3_mask = m_mask[:, None] & n_mask[None, :] & k_mask
        getitem_3 = tl.load(getitem_3_ptr + m_3_offset, mask=offsets_3_mask, other=0.0)
        getitem_3 = getitem_3.to(tl.float32)
        
        # Accumulate
        acc += tl.dot(value_4, getitem_3)
    
    # Convert to output dtype for intermediate operations
    einsum_result = acc.to(tl.float16 if out_ptr.dtype.element_size() == 2 else tl.float32)
    
    # Load out_333 for addition
    out_333_offsets = m_offset[:, None] * H * W + n_offset[None, :]
    out_333_mask = m_mask[:, None] & n_mask[None, :]
    out_333 = tl.load(out_333_ptr + out_333_offsets, mask=out_333_mask, other=0.0).to(tl.float32)
    
    # Fused operations: einsum + add + multiply + add
    # Step 1: in_3 += einsum
    temp = out_333 + einsum_result.to(tl.float32)
    
    # Step 2: multiply by gamma
    temp = temp * gamma
    
    # Step 3: add out_333 (which is in_2 in the original)
    out_333_2_offsets = (m_offset[:, None] * H + c_offset[:, None]) * W + w_offset[:, None]
    out_333_2 = tl.load(out_333_ptr + out_333_2_offsets, mask=out_333_mask, other=0.0).to(tl.float32)
    result = temp + out_333_2
    
    # Store final result in output dtype
    final_result = result.to(tl.float16 if out_ptr.dtype.element_size() == 2 else tl.float32)
    tl.store(out_ptr + out_offsets, final_result, mask=out_mask)

@torch.fx.wrap
def fused_forward_pass(in_0, in_1, in_2, in_3, in_4):
    """Fused forward pass with optimized Triton kernel"""
    B, C, H, W = in_3.shape
    J = in_1.shape[-1]  # Last dimension of in_1
    
    # Determine appropriate block sizes based on tensor dimensions
    if B * C * H * W > 1000000:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64  
        BLOCK_SIZE_K = 32
    elif B * C * H * W > 100000:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    else:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 8
    
    # Calculate grid size
    M = (B * C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    N = (H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(in_3, dtype=in_3.dtype)
    
    # Launch kernel
    fused_kernel[(M, N)](
        value_4_ptr=in_4,
        getitem_3_ptr=in_1,
        out_333_ptr=in_3,
        gamma_params_ptr=in_0,  # Scalar gamma parameter
        out_ptr=out,
        B=B, C=C, H=H, W=W, J=J,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    """Return the fused forward pass function"""
    return fused_forward_pass