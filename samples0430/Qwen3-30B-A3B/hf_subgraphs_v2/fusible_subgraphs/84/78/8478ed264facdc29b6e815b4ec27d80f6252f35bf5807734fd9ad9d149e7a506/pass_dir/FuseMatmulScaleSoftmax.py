import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for matmul, scale, and softmax
@triton.jit
def matmul_scale_softmax_kernel(
    query_ptr, key_ptr, output_ptr,
    batch_size, seq_len_q, seq_len_k, head_size,
    scale: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles a row of the result matrix (batch, seq_len_q, seq_len_k)
    pid = tl.program_id(0)
    block_row = pid // tl.cdiv(seq_len_k, BLOCK_SIZE_N)
    block_col = pid % tl.cdiv(seq_len_k, BLOCK_SIZE_N)
    
    # Create offsets for the current block
    row_start = block_row * BLOCK_SIZE_M
    col_start = block_col * BLOCK_SIZE_N
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (head_size)
    for k in range(0, tl.cdiv(head_size, BLOCK_SIZE_K)):
        # Load query block (BLOCK_SIZE_M x BLOCK_SIZE_K)
        query_block = tl.load(
            query_ptr + (row_start * head_size) + (k * BLOCK_SIZE_K) + (0 * head_size * seq_len_q),
            shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len_q and tl.arange(0, BLOCK_SIZE_K)[None, :] < head_size - k * BLOCK_SIZE_K,
            other=0.0
        )
        # Load key block (BLOCK_SIZE_K x BLOCK_SIZE_N)
        key_block = tl.load(
            key_ptr + (k * BLOCK_SIZE_K * seq_len_k) + (col_start),
            shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < head_size - k * BLOCK_SIZE_K and tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len_k,
            other=0.0
        )
        
        # Perform matrix multiplication (query_block @ key_block)
        acc += tl.dot(query_block, key_block)
    
    # Scale by 1/sqrt(head_size)
    acc = acc * scale
    
    # Compute the max value for softmax to avoid overflow
    max_val = tl.max(acc, axis=1)[:, None]
    
    # Compute exponentials
    exp = tl.exp(acc - max_val)
    
    # Compute the sum for softmax denominator
    sum_exp = tl.sum(exp, axis=1)[:, None]
    
    # Apply softmax
    softmax_out = exp / sum_exp
    
    # Store the result
    tl.store(
        output_ptr + (row_start * seq_len_k) + col_start,
        softmax_out,
        mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len_q and tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len_k
    )

# Kernel wrapper
@torch.fx.wrap
def triton_matmul_scale_softmax(query, key):
    # Get the shapes
    batch_size, num_heads, seq_len_q, head_size = query.shape
    _, _, seq_len_k, _ = key.shape
    
    # Output shape: (batch, num_heads, seq_len_q, seq_len_k)
    output = torch.empty(batch_size, num_heads, seq_len_q, seq_len_k, dtype=query.dtype, device=query.device)
    
    # Calculate the scale factor
    scale = 1.0 / tl.sqrt(head_size)
    
    # Determine block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = tl.cdiv(seq_len_q, BLOCK_SIZE_M)
    grid_n = tl.cdiv(seq_len_k, BLOCK_SIZE_N)
    num_programs = grid_m * grid_n
    
    # Launch the kernel
    matmul_scale_softmax_kernel[(num_programs,)](
        query_ptr=query,
        key_ptr=key,
        output_ptr=output,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_size=head_size,
        scale=scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_matmul_scale_softmax