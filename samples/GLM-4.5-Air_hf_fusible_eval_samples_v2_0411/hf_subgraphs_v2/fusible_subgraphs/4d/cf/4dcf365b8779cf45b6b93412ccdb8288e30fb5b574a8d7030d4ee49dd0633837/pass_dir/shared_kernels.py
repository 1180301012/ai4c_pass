import torch
import triton
import triton.language as tl

# Helper function to apply type conversion if needed
def apply_type_conversion(x):
    """Apply type conversion from bfloat16/float16 to float32"""
    if hasattr(tl, 'cast'):
        return tl.cast(x, tl.float32)
    else:
        return x.to(tl.float32)

@triton.jit
def linear_kernel_shared(
    x_ptr,           # input tensor  
    weight_ptr,      # weight tensor 
    bias_ptr,        # bias tensor
    out_ptr,         # output tensor
    M,               # first dimension (batch*seq or M)
    N,               # output dimension
    K,               # hidden dimension
    dropout_prob: tl.constexpr,  # dropout probability (0.0 means no dropout)
    BLOCK_SIZE_M: tl.constexpr,  # block size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # block size for N dimension
    BLOCK_SIZE_K: tl.constexpr   # block size for K dimension
):
    """Unified Triton kernel for linear operations with optional dropout"""
    pid = tl.program_id(0)
    m_block = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_block < M
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load weight block [BLOCK_SIZE_K, BLOCK_SIZE_N]
        weight_block = tl.load(
            weight_ptr + k_offsets[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N),
            other=0.0
        ).to(tl.float32)
        
        # Load input block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_block = tl.load(
            x_ptr + m_block[:, None] * K + k_offsets[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Apply type conversion if needed
        x_block = apply_type_conversion(x_block)
        
        # Matrix multiplication: acc += x_block @ weight_block.T
        acc += tl.dot(x_block, weight_block)
    
    # Load bias and add to accumulation
    bias_block = tl.load(
        bias_ptr + tl.arange(0, BLOCK_SIZE_N),
        mask=tl.arange(0, BLOCK_SIZE_N) < N,
        other=0.0
    )
    acc += bias_block[None, :]
    
    # Apply dropout if needed (prob > 0)
    if dropout_prob > 0.0:
        dropout_mask = (tl.rand(acc.shape, dtype=tl.float32) > dropout_prob).to(tl.float32)
        acc = acc * dropout_mask
    
    # Store result
    out_block = acc.to(tl.float16)  # Store in float16 to match original precision
    
    # Handle storing with proper masking
    for i in range(BLOCK_SIZE_M):
        if m_block[i] < M:
            n_offsets = tl.arange(0, BLOCK_SIZE_N)
            store_mask = n_offsets < N
            tl.store(
                out_ptr + m_block[i] * N + n_offsets,
                out_block[i, :],
                mask=store_mask
            )

@torch.fx.wrap
def linear_fused_dispatch(x, weight, bias, route_string):
    """Dispatch wrapper that handles different linear operation patterns"""
    
    # Handle different input shapes
    if x.dim() == 3:  # [batch, seq_len, hidden_size]
        M = x.shape[0] * x.shape[1]  # batch_size * seq_len
        N = bias.shape[0]            # output_size
        K = x.shape[2]               # hidden_size
    else:  # [M, K] (already flattened)
        M, K = x.shape
        N = bias.shape[0]
    
    # Output tensor (float16)
    out = torch.empty(M, N, dtype=torch.float16, device=x.device)
    
    # Block sizes for Triton kernel
    BLOCK_SIZE_M = 8   # M dimension granularity
    BLOCK_SIZE_N = 32  # N dimension granularity  
    BLOCK_SIZE_K = 32  # K dimension granularity
    
    # Determine dropout probability and route
    route = route_string.lower()
    if "dropout" in route:
        dropout_prob = 0.1  # For DropoutLinearFusion
    elif "typeconv" in route:
        dropout_prob = 0.0  # For TypeConversionLinearFusion (no dropout)
    else:
        dropout_prob = 0.0  # Default: no dropout
    
    # Launch kernel
    grid = ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, )
    linear_kernel_shared[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M, N=N, K=K,
        dropout_prob=dropout_prob,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Reshape back to original [batch, seq_len, output_size] if needed
    if x.dim() == 3:
        out = out.view(x.shape[0], x.shape[1], N)
    
    return out