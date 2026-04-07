import torch
import triton
import triton.language as tl

# Pattern matching for the matrix multiplication operation
def pattern(in_1, in_3):
    # Original first operation:
    matmul = in_1 @ in_3
        
    # Return the result for the rest of the computation
    return matmul


def replacement_args(in_1, in_3):
    return (in_1, in_3)


@triton.jit
def simple_matmul_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    batch_size, seq1, seq2, heads, seq_len,
    BLOCK_K: tl.constexpr
):
    """
    Optimized matrix multiplication kernel with better memory access patterns
    """
    m_pid = tl.program_id(0)  # batch*seq1*seq2  
    n_pid = tl.program_id(1)  # seq_len
    
    batch = m_pid // (seq1 * seq2)
    seq1_idx = (m_pid % (seq1 * seq2)) // seq2
    seq2_idx = m_pid % seq2
    
    # Initialize accumulator
    acc = 0.0
    
    # Vectorized loop over heads with better memory coalescing
    for k in range(0, heads, BLOCK_K):
        # Load a vector of elements from A
        k_end = min(k + BLOCK_K, heads)
        for kk in range(k, k_end):
            # Load from A: [batch, seq1, seq2, heads]
            a_offset = ((batch * seq1 + seq1_idx) * seq2 + seq2_idx) * heads + kk
            a_val = tl.load(a_ptr + a_offset, mask=kk < heads)
            
            # Load from B: [heads, seq_len]  
            b_offset = kk * seq_len + n_pid
            b_val = tl.load(b_ptr + b_offset, mask=n_pid < seq_len)
            
            acc += a_val * b_val
    
    # Store result to C: [batch, seq1, seq2, seq_len]
    c_offset = ((batch * seq1 + seq1_idx) * seq2 * seq_len + seq2_idx * seq_len + n_pid)
    tl.store(c_ptr + c_offset, acc)


@torch.fx.wrap  
def optimized_matmul(in_1, in_3):
    """
    Optimized matrix multiplication using Triton kernel with adaptive block sizing
    """
    # Get input shapes and handle different tensor dimensions
    a_shape = in_1.shape
    b_shape = in_3.shape
    
    # Handle different tensor dimensions flexibly
    if len(a_shape) == 4:
        # Case 1: [batch, seq1, seq2, heads] @ [heads, seq_len] → [batch, seq1, seq2, seq_len]
        batch_size = a_shape[0]
        seq1 = a_shape[1]
        seq2 = a_shape[2] 
        heads = a_shape[3]
        seq_len = b_shape[1]
        
        # Choose optimal block size based on matrix dimensions
        if heads > 64:
            block_k = 32
        elif heads > 32:
            block_k = 16
        elif heads > 16:
            block_k = 8
        else:
            block_k = 4
            
        # Create output tensor
        output = torch.empty((batch_size, seq1, seq2, seq_len), 
                            dtype=in_1.dtype, 
                            device=in_1.device)
        
        # Always use Triton kernel with optimized configuration
        grid = (batch_size * seq1 * seq2, seq_len)
        
        simple_matmul_kernel[grid](
            a_ptr=in_1,
            b_ptr=in_3,
            c_ptr=output,
            batch_size=batch_size,
            seq1=seq1,
            seq2=seq2,
            heads=heads,
            seq_len=seq_len,
            BLOCK_K=block_k
        )
        
    elif len(a_shape) == 3:
        # Case 2: [batch, seq_len, seq_len] @ [batch, seq_len, heads] → [batch, seq_len, heads]
        # This is the second matmul: tmp_13 @ in_4
        batch_size = a_shape[0]
        seq_len1 = a_shape[1]  # First seq_len (from tmp_13)
        seq_len2 = b_shape[2]  # heads dimension (from in_4)
        
        # Choose optimal block size
        if seq_len1 > 64:
            block_k = 32
        elif seq_len1 > 32:
            block_k = 16
        else:
            block_k = 4
            
        # Create output tensor
        output = torch.empty((batch_size, seq_len1, seq_len2), 
                            dtype=in_1.dtype, 
                            device=in_1.device)
        
        # Always use Triton kernel
        grid = (batch_size * seq_len1, seq_len2)
        
        simple_matmul_kernel[grid](
            a_ptr=in_1,
            b_ptr=in_3,
            c_ptr=output,
            batch_size=batch_size,
            seq1=seq_len1,
            seq2=1,
            heads=1,
            seq_len=seq_len2,
            BLOCK_K=block_k
        )
    
    return output


def replacement_func():
    return optimized_matmul