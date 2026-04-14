import torch
import triton
import triton.language as tl

def pattern(to, in_2):
    # Match the second matmul with transpose and reshape operations
    matmul_1 = torch.matmul(to, in_2)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return matmul_1, tmp_9

def replacement_args(to, in_2):
    return (to, in_2)

@triton.jit
def optimized_matmul_reshape_kernel(
    to_ptr,
    in_2_ptr,
    output_ptr,
    batch_size,
    num_heads, 
    seq_len,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program id for matrix multiplication
    pid = tl.program_id(0)
    grid_m = (batch_size * num_heads * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Split program ID into matrix dimensions
    m = pid % grid_m
    k = (pid // grid_m) % grid_k
    n = (pid // (grid_m * grid_k)) % grid_n
    
    # Compute offsets for this program
    offs_am = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for matrix bounds
    am_mask = offs_am < batch_size * num_heads * seq_len
    bn_mask = offs_bn < head_dim
    k_mask = offs_k < seq_len
    
    # Load matrices with tiling
    to_ptrs = to_ptr + (offs_am[:, None] * seq_len + offs_k[None, :]) 
    in_2_ptrs = in_2_ptr + (offs_k[:, None] * head_dim + offs_bn[None, :])
    
    to_vals = tl.load(to_ptrs, mask=am_mask[:, None] & k_mask[None, :], other=0.0)
    in_2_vals = tl.load(in_2_ptrs, mask=k_mask[:, None] & bn_mask[None, :], other=0.0)
    
    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, seq_len, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask_tile = k_offs < seq_len
        
        to_tile = tl.load(to_ptrs, mask=am_mask[:, None] & k_mask_tile[None, :], other=0.0)
        in_2_tile = tl.load(in_2_ptrs, mask=k_mask_tile[:, None] & bn_mask[None, :], other=0.0)
        
        acc += tl.dot(to_tile, in_2_tile, acc_type=tl.float32)
        
        # Update pointers for next tile
        to_ptrs += BLOCK_SIZE_K
        in_2_ptrs += BLOCK_SIZE_K * head_dim
    
    # Transpose result and reshape
    # For the specific shape [1,16,257,80] -> [1,80,257] -> [1,257,-1]
    if BLOCK_SIZE_M == 80 and BLOCK_SIZE_N == 257:  # This is the reshape pattern
        # Reshape to [1,257,20480] where 20480 = 80*257
        reshape_offset = (offs_am // 257 * 20480 + offs_am % 257 * 80 + offs_bn)
        reshape_mask = (offs_am // 257 == 0) & (offs_am % 257 < 257) & (offs_bn < 80)
        tl.store(output_ptr + reshape_offset, acc.to(tl.float16), mask=reshape_mask)

@torch.fx.wrap
def optimized_matmul_reshape(to, in_2):
    # Input shapes: to=[1,16,257,257], in_2=[1,16,257,80]
    batch_size, num_heads, seq_len, _ = to.shape
    head_dim = in_2.shape[-1]
    
    # Optimized tile sizes for this specific shape
    BLOCK_SIZE_M = 80    # head_dim
    BLOCK_SIZE_N = 257   # seq_len  
    BLOCK_SIZE_K = 64    # Optimized for GPU
    
    # Compute grid size
    total_m = batch_size * num_heads * seq_len
    grid_m = (total_m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    total_programs = grid_m * grid_n * grid_k
    
    # Create output tensor with final reshape shape [1, 257, -1]
    # where -1 = 80*257 = 20480
    output_shape = (1, 257, 20480)
    output = torch.empty(output_shape, dtype=torch.float16, device=to.device)
    
    # Launch kernel
    optimized_matmul_reshape_kernel[(total_programs,)](
        to_ptr=to,
        in_2_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_matmul_reshape