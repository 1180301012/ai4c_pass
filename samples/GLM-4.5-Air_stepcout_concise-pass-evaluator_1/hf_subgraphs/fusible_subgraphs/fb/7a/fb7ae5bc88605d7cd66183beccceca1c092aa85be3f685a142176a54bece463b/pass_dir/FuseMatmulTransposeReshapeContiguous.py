import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching the main computation bottleneck: softmax + matmul + transformations
    """
    # This captures the core computation that can be optimized
    tmp_0 = in_0 * 1.0  # Multiplication by 1.0 (no-op but included for exact matching)
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1, dtype=torch.float32)
    tmp_0 = None
    tmp_2 = tmp_1.to(torch.float32)
    tmp_1 = None
    tmp_3 = torch.nn.functional.dropout(tmp_2, p=0.0, training=False)
    tmp_2 = None
    
    # Main computation: matmul
    tmp_4 = torch.matmul(tmp_3, in_1)
    tmp_3 = None
    
    # Final transformations
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_4 = None
    tmp_6 = tmp_5.contiguous()
    tmp_5 = None
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_6 = None
    tmp_8 = tmp_7.contiguous()
    tmp_7 = None
    
    return tmp_8

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_attention_kernel(
    query_ptr,           # [1, 16, 257, 257] - input query after scaling
    key_ptr,             # [1, 16, 257, 80]  - key matrix
    output_ptr,          # [1, 257, 1280]    - final output
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len_q: tl.constexpr,
    seq_len_k: tl.constexpr,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Fused kernel for attention computation with final reshape"""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)  # batch dimension
    
    # Process sequence positions
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m < seq_len_q
    
    for h in range(num_heads):
        # Calculate offsets for current head
        query_offset = pid_b * num_heads * seq_len_q * seq_len_k + h * seq_len_q * seq_len_k + m[:, None] * seq_len_k
        key_offset = pid_b * num_heads * seq_len_k * embedding_dim + h * seq_len_k * embedding_dim
        
        # For final output, we want [1, seq_len_q, num_heads * embedding_dim]
        final_output_offset = pid_b * seq_len_q * (num_heads * embedding_dim) + m[:, None] * (num_heads * embedding_dim) + h * embedding_dim
        
        # Load query values [BLOCK_SIZE_M, seq_len_k]
        query_vals = tl.load(query_ptr + query_offset, mask=m_mask[:, None], other=0.0)
        
        # Apply softmax along last dimension (keys dimension)
        max_vals = tl.max(query_vals, axis=1, keepdims=True)
        exp_vals = tl.exp(query_vals - max_vals)
        sum_vals = tl.sum(exp_vals, axis=1, keepdims=True)
        softmax_vals = exp_vals / sum_vals
        
        # Load key values [seq_len_k, embedding_dim]
        keys = tl.zeros((seq_len_k, embedding_dim), dtype=tl.float32)
        for k in range(0, seq_len_k, 32):
            block_size = min(32, seq_len_k - k)
            key_block = tl.load(
                key_ptr + key_offset + k * embedding_dim,
                mask=(tl.arange(0, block_size)[:, None] < block_size) & 
                     (tl.arange(0, embedding_dim)[None, :] < embedding_dim),
                other=0.0
            )
            keys[k:k+block_size, :] = key_block
        
        # Matrix multiplication: [BLOCK_SIZE_M, seq_len_k] @ [seq_len_k, embedding_dim]
        acc = tl.zeros((BLOCK_SIZE_M, embedding_dim), dtype=tl.float32)
        for k in range(seq_len_k):
            if k < seq_len_k:
                key_vals = keys[k, :]
                # Outer product with broadcasting
                acc += softmax_vals[:, k:k+1] * key_vals[None, :]
        
        # Store result directly in final flattened format
        tl.store(output_ptr + final_output_offset, acc, mask=m_mask[:, None])

@torch.fx.wrap
def fused_attention_forward(in_0, in_1):
    """
    Optimized forward pass that fuses:
    1. Softmax attention computation
    2. Matmul with keys
    3. Transpose and reshape to final format [1, 257, 1280]
    """
    # Extract tensor dimensions
    batch_size = in_0.shape[0]  # 1
    num_heads = in_0.shape[1]   # 16
    seq_len_q = in_0.shape[2]   # 257
    seq_len_k = in_0.shape[3]   # 257
    embedding_dim = in_1.shape[3]  # 80
    
    # Final output shape: [1, 257, 1280] (16*80=1280)
    output_shape = [batch_size, seq_len_q, num_heads * embedding_dim]
    output = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 32  # Number of sequence positions to process per program
    num_blocks_m = (seq_len_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch the fused kernel
    fused_attention_kernel[
        (num_blocks_m, batch_size),
    ](
        in_0,
        in_1,
        output,
        batch_size,
        num_heads,
        seq_len_q,
        seq_len_k,
        embedding_dim,
        BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    return fused_attention_forward