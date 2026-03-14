import torch
import triton
import triton.language as tl

def pattern(tmp_0, tmp_1, in_2, in_3):
    # tmp_0 is bias, tmp_1 is weight, in_2 is input, in_3 is residual
    tmp_3 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    tmp_5 = in_3 + tmp_4
    return tmp_5, tmp_4

def replacement_args(tmp_0, tmp_1, in_2, in_3):
    return (tmp_0, tmp_1, in_2, in_3)

@triton.jit
def fused_linear_dropout_add_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    residual_ptr,
    out_ptr,
    dropout_out_ptr,
    n_batch,
    n_hidden,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Load bias once and broadcast
    bias = tl.load(bias_ptr)
    
    # Compute program IDs for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create range of offsets for M and N dimensions
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundary conditions
    m_mask = m_offsets < n_batch
    n_mask = n_offsets < n_hidden
    
    # Load input chunks (transposed for efficient matmul)
    input_offsets = (m_offsets[:, None] * n_hidden + tl.arange(0, BLOCK_SIZE_K)[None, :]).to(tl.int64)
    input_chunk = tl.load(input_ptr + input_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Load weight chunks
    weight_offsets = (n_offsets[None, :] * n_hidden + tl.arange(0, BLOCK_SIZE_K)[:, None]).to(tl.int64)
    weight_chunk = tl.load(weight_ptr + weight_offsets, mask=n_mask[None, :] & n_mask[:, None], other=0.0)
    
    # Compute linear transformation: input @ weight.T + bias
    # For [batch, hidden] @ [hidden, hidden] -> [batch, hidden]
    # We need to transpose the weight in memory or adjust our indexing
    linear_result = tl.dot(input_chunk, weight_chunk.T, allow_tf32=False) + bias
    
    # Since dropout p=0.0, just copy the result
    dropout_result = linear_result
    
    # Load residual for addition
    residual_offsets = (m_offsets[:, None] * n_hidden + n_offsets[None, :]).to(tl.int64)
    residual_chunk = tl.load(residual_ptr + residual_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Add residual
    out_result = linear_result + residual_chunk
    
    # Store results
    out_offsets = (m_offsets[:, None] * n_hidden + n_offsets[None, :]).to(tl.int64)
    tl.store(out_ptr + out_offsets, out_result, mask=m_mask[:, None] & n_mask[None, :])
    
    # Store dropout result (no-op in this case since p=0.0)
    tl.store(dropout_out_ptr + out_offsets, dropout_result, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_linear_dropout_add(tmp_0, tmp_1, in_2, in_3):
    n_batch = in_2.shape[0]
    n_hidden = in_2.shape[1]
    
    # Choose optimal block sizes for typical sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    
    # Calculate number of programs
    num_m = (n_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n = (n_hidden + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensors
    out = torch.empty_like(in_2)
    dropout_out = torch.empty_like(in_2)
    
    # Launch kernel
    fused_linear_dropout_add_kernel[(num_m, num_n)](
        bias_ptr=tmp_0,
        weight_ptr=tmp_1,
        input_ptr=in_2,
        residual_ptr=in_3,
        out_ptr=out,
        dropout_out_ptr=dropout_out,
        n_batch=n_batch,
        n_hidden=n_hidden,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out, dropout_out

def replacement_func():
    return fused_linear_dropout_add