import torch
import triton
import triton.language as tl

def pattern(tmp_0, tmp_1, in_3, in_2):
    # tmp_0 is bias, tmp_1 is weight, in_3 is input, in_2 is residual
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    tmp_4 = tmp_3 + in_2
    return tmp_4,

def replacement_args(tmp_0, tmp_1, in_3, in_2):
    return (tmp_0, tmp_1, in_3, in_2)

@triton.jit
def fused_linear_dropout_add_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    residual_ptr,
    out_ptr,
    n_batch,
    n_hidden,
    dropout_p: tl.constexpr,
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
    
    # Load input chunks
    input_offsets = (m_offsets[:, None] * n_hidden + tl.arange(0, BLOCK_SIZE_K)[None, :]).to(tl.int64)
    input_chunk = tl.load(input_ptr + input_offsets, mask=m_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < n_hidden), other=0.0)
    
    # Load weight chunks (transposed for efficient computation)
    weight_offsets = (tl.arange(0, BLOCK_SIZE_K)[:, None] * n_hidden + n_offsets[None, :]).to(tl.int64)
    weight_chunk = tl.load(weight_ptr + weight_offsets, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < n_hidden) & n_mask[None, :], other=0.0)
    
    # Compute linear transformation: input @ weight.T + bias
    linear_result = tl.dot(input_chunk, weight_chunk, allow_tf32=False) + bias
    
    # Apply dropout using random number generation
    # Generate random numbers for dropout
    rand_offset = tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N)
    rng_state = tl.rand(rand_offset.to(tl.uint32))
    mask = rng_state > dropout_p
    
    # Reshape mask to match linear_result shape
    dropout_mask = mask.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
    dropout_mask = dropout_mask & m_mask[:, None] & n_mask[None, :]
    
    # Apply dropout
    dropout_result = linear_result * dropout_mask
    
    # Load residual for addition
    residual_offsets = (m_offsets[:, None] * n_hidden + n_offsets[None, :]).to(tl.int64)
    residual_chunk = tl.load(residual_ptr + residual_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Add residual
    out_result = dropout_result + residual_chunk
    
    # Store result
    out_offsets = (m_offsets[:, None] * n_hidden + n_offsets[None, :]).to(tl.int64)
    tl.store(out_ptr + out_offsets, out_result, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_linear_dropout_add(tmp_0, tmp_1, in_3, in_2):
    n_batch = in_3.shape[0]
    n_hidden = in_3.shape[1] if len(in_3.shape) > 1 else in_3.shape[-1]
    
    # Choose optimal block sizes based on input dimensions
    if n_batch >= 64 and n_hidden >= 128:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
    elif n_batch >= 32 and n_hidden >= 64:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64  
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    
    # Calculate number of programs
    num_m = (n_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n = (n_hidden + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Launch kernel with dropout probability 0.1
    fused_linear_dropout_add_kernel[(num_m, num_n)](
        bias_ptr=tmp_0,
        weight_ptr=tmp_1,
        input_ptr=in_3,
        residual_ptr=in_2,
        out_ptr=out,
        n_batch=n_batch,
        n_hidden=n_hidden,
        dropout_p=0.1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return fused_linear_dropout_add