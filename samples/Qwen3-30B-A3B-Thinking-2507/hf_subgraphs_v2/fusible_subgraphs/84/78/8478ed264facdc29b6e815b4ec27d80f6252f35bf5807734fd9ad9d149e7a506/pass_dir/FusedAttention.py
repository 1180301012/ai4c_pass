import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1, in_2):
    # Compute query-key dot product
    matmul = torch.matmul(in_0, in_1)
    # Scale the dot product
    scaled = matmul / 5.656854249492381
    # Apply softmax
    tmp_2 = torch.nn.functional.softmax(scaled, dim=-1)
    # Apply dropout (no-op for 0.0)
    dropout = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    # Compute context
    tmp_4 = torch.matmul(dropout, in_2)
    return tmp_2, tmp_4
    return attn_probs, context

# Argument extraction function

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel
@triton.jit

def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr,
    batch_size, heads, seq_len_q, seq_len_k, dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Calculate the block index
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # Calculate offsets for the current block
    block_start_q = block_id * BLOCK_SIZE_Q
    block_start_k = 0

    # Calculate offsets for memory loads
    q_offset = (batch_id * heads + head_id) * seq_len_q * dim + block_start_q * dim
    k_offset = (batch_id * heads + head_id) * seq_len_k * dim
    v_offset = (batch_id * heads + head_id) * seq_len_k * dim
    out_offset = (batch_id * heads + head_id) * seq_len_q * seq_len_k + block_start_q * seq_len_k

    # Allocate space in shared memory for Q and K
    q_shared = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    k_shared = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

    # Load Q block into shared memory
    for i in range(0, seq_len_q, BLOCK_SIZE_Q):
        q_offsets = q_offset + i * dim
        q_block = tl.load(q_ptr + q_offsets, mask=(i + tl.arange(0, BLOCK_SIZE_Q) < seq_len_q)[:, None],
                        other=0.0)
        q_shared = tl.where(q_block == 0.0, q_shared, q_block)

    # Load K block into shared memory
    for i in range(0, seq_len_k, BLOCK_SIZE_K):
        k_offsets = k_offset + i * dim
        k_block = tl.load(k_ptr + k_offsets, mask=(i + tl.arange(0, BLOCK_SIZE_K) < seq_len_k)[:, None],
                        other=0.0)
        k_shared = tl.where(k_block == 0.0, k_shared, k_block)

    # Compute attention scores
    scores = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    for d in range(0, dim, BLOCK_SIZE_D):
        # Load Q and K tiles
        q_tile = tl.load(q_shared[:, :BLOCK_SIZE_D], mask=tl.arange(0, BLOCK_SIZE_Q)[:, None] < (BLOCK_SIZE_Q),
                        other=0.0)
        k_tile = tl.load(k_shared[:, :BLOCK_SIZE_D], mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < (BLOCK_SIZE_K),
                        other=0.0)
        
        # Compute dot product
        dprod = tl.dot(q_tile, k_tile.T)
        scores += dprod

    # Apply softmax
    # We're doing softmax over the key dimension (dim=-1)
    scores = scores / tl.sqrt(float(dim))
    exp_scores = tl.exp(scores - tl.max(scores, axis=1, keepdims=True))
    sum_exp = tl.sum(exp_scores, axis=1, keepdims=True)
    attn_probs = exp_scores / sum_exp

    # Compute output (context)
    context = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    for d in range(0, dim, BLOCK_SIZE_D):
        # Load V block
        v_offsets = v_offset + block_start_k * dim + d
        v_block = tl.load(v_ptr + v_offsets, mask=(block_start_k + tl.arange(0, BLOCK_SIZE_K) < seq_len_k)[:, None],
                        other=0.0)
        
        # Compute context
        context += tl.dot(attn_probs, v_block)

    # Store output
    tl.store(out_ptr + out_offset, context, mask=(block_start_q + tl.arange(0, BLOCK_SIZE_Q) < seq_len_q)[:, None] &
                                              (block_start_k + tl.arange(0, BLOCK_SIZE_K) < seq_len_k)[None, :])

# Kernel wrapper
@torch.fx.wrap

def fused_attention_wrapper(q, k, v):
    batch_size, heads, seq_len_q, dim = q.shape
    _, _, seq_len_k, _ = k.shape

    # Calculate block sizes based on dimension
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = 64

    # Create output tensor
    out = torch.empty((batch_size, heads, seq_len_q, seq_len_k), device=q.device, dtype=q.dtype)

    # Calculate grid dimensions
    grid = (batch_size, heads, (seq_len_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q)

    # Launch the kernel
    fused_attention_kernel[grid](
        q_ptr=q, k_ptr=k, v_ptr=v,
        out_ptr=out,
        batch_size=batch_size,
        heads=heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        dim=dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out

# Replacement function

def replacement_func():
    return fused_attention_wrapper