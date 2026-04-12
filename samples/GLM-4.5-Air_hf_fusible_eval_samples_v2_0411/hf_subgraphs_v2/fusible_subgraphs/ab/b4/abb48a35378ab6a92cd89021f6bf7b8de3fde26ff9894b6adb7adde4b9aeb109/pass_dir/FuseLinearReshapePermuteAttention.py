import torch
import triton
import triton.language as tl

def pattern(weight, input_tensor):
    """Pattern matches linear + reshape + permute for attention mechanism"""
    tmp_0 = weight
    tmp_1 = torch.nn.functional.linear(input_tensor, tmp_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, -1, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    return tmp_3

@triton.jit
def attention_qkv_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    input_batch_size,
    input_seq_len,
    input_features,
    output_total_dim,
    n_heads,
    head_dim,
    weight_rows,
    weight_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for fused QKV computation"""
    # Matrix multiplication program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges each program should process
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, input_batch_size * input_seq_len)
    
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, output_total_dim)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    k = tl.arange(0, BLOCK_SIZE_K)
    weight_ptrs = weight_ptr + weight_cols * k[:, None] + n_start[None, :]
    input_ptrs = input_ptr + input_features * k[None, :] + (m_start % (input_batch_size * input_seq_len))[None, :]
    
    for k_block_start in range(0, weight_cols, BLOCK_SIZE_K):
        k_end = min(k_block_start + BLOCK_SIZE_K, weight_cols)
        
        # Load weights and inputs
        mask_weight = k_block_start + k < weight_cols
        mask_input = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < m_end
        
        weight_vals = tl.load(weight_ptrs, mask=mask_weight[:, None], other=0.0)
        input_vals = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        # Compute matrix multiplication block
        accumulator += tl.dot(input_vals, weight_vals)
        
        # Update pointers for next iteration
        weight_ptrs += weight_cols * (k_end - k_block_start)
        input_ptrs += input_features * (k_end - k_block_start)
    
    # Store result
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    mask = (m_offsets[:, None] < m_end) & (n_offsets[None, :] < n_end)
    
    output_base_ptr = output_ptr + (m_offsets[:, None] * output_total_dim + n_offsets[None, :])
    tl.store(output_base_ptr, accumulator, mask=mask)

@torch.fx.wrap
def fused_attention_qkv_linear(weight, input_tensor):
    """Fused linear + reshape + permute for attention QKV computation"""
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    input_features = input_tensor.shape[2]
    
    weight_rows = weight.shape[0]
    weight_cols = weight.shape[1]
    output_total_dim = weight_rows  # This should be 1296, 3*9*48
    
    # Calculate optimal block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Launch grid dimensions
    grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_total_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output tensor
    linear_out = torch.empty(batch_size, seq_len, output_total_dim, 
                           dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    attention_qkv_kernel[(grid_m, grid_n)](
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=linear_out,
        input_batch_size=batch_size,
        input_seq_len=seq_len,
        input_features=input_features,
        output_total_dim=output_total_dim,
        n_heads=3,
        head_dim=None,
        weight_rows=weight_rows,
        weight_cols=weight_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape and permute (these are cheap view operations)
    reshaped = linear_out.reshape(1, 197, 3, -1, 48)
    permuted = reshaped.permute(2, 0, 3, 1, 4)
    return permuted

def replacement_args(weight, input_tensor):
    """Extract arguments for the fused attention kernel"""
    return (weight, input_tensor)

def replacement_func():
    """Return the fused attention function"""
    return fused_attention_qkv_linear