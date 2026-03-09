import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(1, 1, 1, 1)  # Simple placeholder that matches the view pattern
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_matmul_view_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size,
    dim1,
    M,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Each program handles one batch * dim1 pair
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    batch_offset = pid_b * (dim1 * M)
    dim1_offset = pid_d * M
    m_offset = pid_m * BLOCK_SIZE_M
    
    in_0_batch_offset = pid_b * (dim1 * K * 1)
    in_1_batch_offset = pid_b * (dim1 * M * K)
    
    in_0_dim1_offset = pid_d * (K * 1)
    in_1_dim1_offset = pid_d * (M * K)
    
    # Compute pointers for this M block
    in_0_base_ptr = in_0_ptr + in_0_batch_offset + in_0_dim1_offset
    in_1_base_ptr = in_1_ptr + in_1_batch_offset + in_1_dim1_offset
    out_base_ptr = out_ptr + batch_offset + dim1_offset + m_offset
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        bounds_k = k + BLOCK_SIZE_K
        
        # Load in_0 [K, 1] -> same for all M elements in block
        in_0_offset = k * 1
        in_0_val = tl.load(in_0_base_ptr + in_0_offset, mask=(k < K), other=0.0)
        
        # Load in_1 slice [M_BLOCK_SIZE, K_BLOCK_SIZE]
        in_1_offsets = (tl.arange(0, BLOCK_SIZE_M)[:, None] * K + 
                       (tl.arange(0, BLOCK_SIZE_K)[None, :]))
        in_1_mask = ((tl.arange(0, BLOCK_SIZE_M)[:, None] + m_offset < M) & 
                    (tl.arange(0, BLOCK_SIZE_K)[None, :] < bounds_k))
        in_1_vals = tl.load(in_1_base_ptr + in_1_offsets, mask=in_1_mask, other=0.0)
        
        # Matmul: accumulate
        accumulator += tl.dot(in_1_vals, in_0_val, out_layout=zM)
    
    # Store result with view applied - already in [B, D, M, 1] shape
    out_mask = (tl.arange(0, BLOCK_SIZE_M) + m_offset < M)
    tl.store(out_base_ptr, accumulator, mask=out_mask[:, None])

@torch.fx.wrap
def fused_matmul_view(in_0, in_1):
    B, D1, K, _ = in_0.shape
    M = in_1.shape[2]
    
    # Output shape is [B, D1, M, 1]
    out_shape = (B, D1, M, 1)
    out = torch.empty(out_shape, dtype=tl.float32, device=in_0.device)
    
    # Set up grid dimensions
    Batch_Dim1_Programs = B * D1
    
    # Configure block sizes based on tensor dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 1  
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Number of programs needed for M dimension
    num_programs_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    fused_matmul_view_kernel[(Batch_Dim1_Programs, num_programs_m, 1)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=B,
        dim1=D1,
        M=M,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return out

def replacement_func():
    return fused_matmul_view