import torch
import triton
import triton.language as tl

# Pattern matching function
@torch.fx.wrap
def pattern(in_0, in_1, in_2):
    x = torch.bmm(in_0, in_1)
    y = torch.nn.functional.softmax(x, dim=-1)
    z = torch.nn.functional.dropout(y, p=0.0, training=False)
    w = torch.bmm(z, in_2)
    return w

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused attention
@triton.jit
def fused_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    n_heads,
    seq_len_q,
    seq_len_kv,
    d_head,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread index
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Compute starting positions
    q_offset = batch_idx * n_heads * seq_len_q * d_head + head_idx * seq_len_q * d_head
    k_offset = batch_idx * n_heads * seq_len_kv * d_head + head_idx * seq_len_kv * d_head
    v_offset = batch_idx * n_heads * seq_len_kv * d_head + head_idx * seq_len_kv * d_head
    out_offset = batch_idx * n_heads * seq_len_q * d_head + head_idx * seq_len_q * d_head

    # Allocate shared memory for scores
    scores = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Compute attention scores (QK^T)
    for s in range(0, seq_len_q, BLOCK_SIZE):
        for t in range(0, seq_len_kv, BLOCK_SIZE):
            # Load Q and K
            q = tl.load(
                q_ptr + q_offset + s * d_head + tl.arange(0, BLOCK_SIZE),
                mask=(s + tl.arange(0, BLOCK_SIZE)) < seq_len_q,
                other=0.0
            )
            k = tl.load(
                k_ptr + k_offset + t * d_head + tl.arange(0, BLOCK_SIZE),
                mask=(t + tl.arange(0, BLOCK_SIZE)) < seq_len_kv,
                other=0.0
            )
            
            # Compute scores (QK^T)
            dot = tl.dot(q, tl.trans(k))
            scores = tl.where(
                (s + tl.arange(0, BLOCK_SIZE).reshape(-1, 1)) < seq_len_q,
                tl.where(
                    (t + tl.arange(0, BLOCK_SIZE)) < seq_len_kv,
                    scores + dot,
                    scores
                ),
                scores
            )

    # Apply softmax
    max_scores = tl.max(scores, axis=1, keepdims=True)
    exp_scores = tl.exp(scores - max_scores)
    sum_exp = tl.sum(exp_scores, axis=1, keepdims=True)
    attn_weights = exp_scores / sum_exp

    # Compute output (attn_weights * V)
    out = tl.zeros((BLOCK_SIZE, d_head), dtype=tl.float32)
    for t in range(0, seq_len_kv, BLOCK_SIZE):
        v = tl.load(
            v_ptr + v_offset + t * d_head + tl.arange(0, BLOCK_SIZE),
            mask=(t + tl.arange(0, BLOCK_SIZE)) < seq_len_kv,
            other=0.0
        )
        out += tl.dot(attn_weights[:, t:t + BLOCK_SIZE], v)

    # Store output
    tl.store(
        out_ptr + out_offset + s * d_head + tl.arange(0, BLOCK_SIZE),
        out,
        mask=(s + tl.arange(0, BLOCK_SIZE)) < seq_len_q
    )

# Wrapper function
@torch.fx.wrap
def fused_attention(in_0, in_1, in_2):
    # Extract shapes
    batch_size = in_0.shape[0]
    n_heads = in_0.shape[1]
    seq_len_q = in_0.shape[2]
    seq_len_kv = in_1.shape[2]
    d_head = in_0.shape[3]

    # Initialize output
    out = torch.empty_like(in_0)

    # Set kernel parameters
    BLOCK_SIZE = 16
    grid = (batch_size, n_heads)

    # Launch kernel
    fused_attn_kernel[grid](
        in_0, in_1, in_2,
        out,
        n_heads,
        seq_len_q,
        seq_len_kv,
        d_head,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Replacement function
def replacement_func():
    return fused_attention