import torch
import triton
import triton.language as tl

def pattern(matmul, in_4):
    # Match the softmax + matmul sequence
    tmp_13 = matmul.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    return matmul_1

def replacement_args(matmul, in_4):
    return (matmul, in_4)

@triton.jit
def fused_softmax_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID for matrix multiplication
    pid = tl.program_id(0)
    block_size = BLOCK_M * BLOCK_N
    
    # Calculate batch and sequence position
    batch_id = pid // (seq_len // BLOCK_M)
    seq_start = (pid % (seq_len // BLOCK_M)) * BLOCK_M
    seq_end = min(seq_start + BLOCK_M, seq_len)
    
    if batch_id >= batch_size or seq_start >= seq_len:
        return
    
    # Compute Softmax + Matmul in a single kernel
    # Load attention scores for this batch and position
    scores_offset = batch_id * seq_len * seq_len + seq_start * seq_len
    max_score = -tl.float32('inf')
    
    # Find max for softmax stability (simplified - would need proper reduction)
    for k in range(seq_len):
        score_val = tl.load(input_ptr + scores_offset + k * seq_len + seq_start, mask=(k < seq_len) & (seq_start < seq_len))
        max_score = tl.maximum(max_score, score_val)
    
    # Compute softmax and multiply with value
    sum_val = 0.0
    for k in range(seq_len):
        score_val = tl.load(input_ptr + scores_offset + k * seq_len + seq_start, mask=(k < seq_len) & (seq_start < seq_len))
        exp_val = tl.exp(score_val - max_score)
        sum_val += exp_val
    
    for k in range(seq_len):
        score_val = tl.load(input_ptr + scores_offset + k * seq_len + seq_start, mask=(k < seq_len) & (seq_start < seq_len))
        exp_val = tl.exp(score_val - max_score)
        softmax_val = exp_val / sum_val
        
        # Multiply with value vector
        for d in range(head_dim):
            val_offset = batch_id * seq_len * head_dim + seq_start * head_dim + d
            weight_offset = (k // (seq_len // BLOCK_K)) * BLOCK_K * head_dim + d
            val = tl.load(weight_ptr + val_offset, mask=d < head_dim)
            result = softmax_val * val
            
            # Store result
            output_offset = batch_id * seq_len * head_dim + seq_start * head_dim + d
            tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_softmax_matmul(matmul, in_4):
    batch_size, seq_len, hidden_dim = matmul.shape
    value_dim = in_4.shape[-1]
    
    output = torch.zeros((batch_size, seq_len, value_dim), dtype=matmul.dtype, device=matmul.device)
    
    # Launch Triton kernel
    grid_size = (batch_size * (seq_len + 127) // 128,)
    fused_softmax_matmul_kernel[grid_size](
        matmul,
        in_4,
        output,
        batch_size,
        seq_len,
        value_dim,
        BLOCK_M=128,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    
    return output

def replacement_func():
    return fused_softmax_matmul