import torch
import triton
import triton.language as tl

@triton.jit
def fused_scale_add_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    scale_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for fused scaling and addition: out = x * scale_factor + mask"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: scaling + addition
    out = x * scale_factor + mask_val
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def optimized_softmax_kernel(
    input_ptr,
    output_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized softmax kernel using Triton"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input for current head/batch position
    input_val = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for stability
    max_val = tl.max(input_val, axis=0)
    
    # Compute exponential
    exp_val = tl.exp(input_val - max_val)
    
    # Compute sum for normalization
    sum_val = tl.sum(exp_val, axis=0)
    
    # Normalize
    softmax_val = exp_val / (sum_val + 1e-20)  # Add small epsilon for stability
    
    tl.store(output_ptr + offsets, softmax_val, mask=mask)

@triton.jit
def optimized_matmul_kernel(
    query_ptr,
    key_ptr,
    output_ptr,
    query_stride,
    key_stride,
    output_stride,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized matrix multiplication kernel using Triton"""
    # Program ID for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create pointers for current block
    query_ptr += m_offset * query_stride[0]
    key_ptr += n_offset * key_stride[0] 
    output_ptr += m_offset * output_stride[0] + n_offset
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k_pos in range(0, k, BLOCK_SIZE_K):
        # Load current blocks
        query_block = tl.load(query_ptr + k_pos, mask=(k_pos < k), other=0.0)
        key_block = tl.load(key_ptr + k_pos * key_stride[1], mask=(k_pos < k), other=0.0)
        
        # Matrix multiplication
        accumulator += query_block[:, :, None] * key_block[None, :, :]
    
    # Store result
    output_ptr += tl.arange(0, BLOCK_SIZE_N)[None, :]
    tl.store(output_ptr, accumulator)

@torch.fx.wrap
def fused_scale_add(x, mask, scale_factor):
    """Fused scaling and addition using Triton"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_scale_add_kernel[(num_programs,)](
        x_ptr=x,
        mask_ptr=mask,
        out_ptr=out,
        scale_factor=scale_factor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def optimized_softmax(x, dim=-1):
    """Optimized softmax using Triton"""
    # Handle different tensor shapes - reshape to 2D for softmax
    if x.dim() == 4:
        # Reshape [B, H, N, N] to [B*H*N, N] for softmax along last dimension
        original_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        out_reshaped = torch.empty_like(x_reshaped)
        
        # Apply softmax to each row
        for i in range(x_reshaped.shape[0]):
            row = x_reshaped[i:i+1, :]
            n_elements = row.numel()
            BLOCK_SIZE = 1024
            num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out_row = torch.empty_like(row)
            optimized_softmax_kernel[(num_programs,)](
                input_ptr=row,
                output_ptr=out_row,
                stride=row.stride(0),
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            out_reshaped[i:i+1, :] = out_row
        
        out = out_reshaped.reshape(original_shape)
    else:
        # Fallback for other shapes
        out = torch.softmax(x, dim=dim)
    
    return out

@torch.fx.wrap
def optimized_attention_softmax_matmul(softmax_output, value):
    """Optimized attention computation - matmul with value matrix"""
    # Reshape for matmul: [B, H, N, N] @ [B, H, N, V] -> [B, H, N, V]
    if softmax_output.dim() == 4 and value.dim() == 4:
        B, H, N, _ = softmax_output.shape
        _, _, _, V_dim = value.shape
        
        # Reshape to 2D for matmul: [B*H*N, N] @ [B*H*N, V] -> [B*H*N, V]
        query_2d = softmax_output.reshape(-1, N)
        key_2d = value.reshape(-1, V_dim)
        
        out_2d = torch.empty(B*H*N, V_dim, dtype=torch.bfloat16, device=value.device)
        
        # Blocked matrix multiplication
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
        
        grid_m = (query_2d.shape[0] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (key_2d.shape[1] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        if grid_m > 0 and grid_n > 0:
            optimized_matmul_kernel[(grid_m, grid_n)](
                query_ptr=query_2d,
                key_ptr=key_2d.T,  # Transpose for matmul
                output_ptr=out_2d,
                query_stride=query_2d.stride(),
                key_stride=key_2d.T.stride(),
                output_stride=out_2d.stride(),
                m=query_2d.shape[0],
                n=key_2d.shape[1],
                k=query_2d.shape[1],
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
            )
        
        # Reshape back to 4D
        out = out_2d.reshape(B, H, N, V_dim)
    else:
        # Fallback to regular matmul
        out = torch.matmul(softmax_output, value)
    
    return out

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the attention computation pattern:
    scaling -> addition -> softmax -> dropout (no-op) -> matmul -> permute -> contiguous
    """
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    
    # Return observable outputs - these are the values used outside the pattern
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the replacement function"""
    return (in_0, in_2, in_3)

def replacement_func():
    """Return the optimized attention computation function"""
    def optimized_attention(in_0, in_2, in_3):
        # Fused scaling + addition (dropout with training=False is removed)
        scaled_masked = fused_scale_add(in_0, in_2, 1.0/8.0)
        
        # Optimized softmax
        softmax_output = optimized_softmax(scaled_masked, dim=-1)
        
        # Optimized matmul with value
        attention_output = optimized_attention_softmax_matmul(softmax_output, in_3)
        
        # Permute and make contiguous
        output = attention_output.permute(0, 2, 1, 3).contiguous()
        
        return (output,)
    
    return optimized_attention