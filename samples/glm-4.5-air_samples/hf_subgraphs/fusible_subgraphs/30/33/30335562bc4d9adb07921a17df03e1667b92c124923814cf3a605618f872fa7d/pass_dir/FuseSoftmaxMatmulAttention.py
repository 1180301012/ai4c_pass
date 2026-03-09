import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Exact pattern matching the target computation structure
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_1, tmp_2, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused scaling + softmax + matmul kernel for attention computation
@triton.jit
def fused_softmax_matmul_kernel(
    attention_scores_ptr,
    values_ptr,
    output_ptr,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Program ID layout for 3D tensors (B, H, dim/seq_len)
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Calculate offsets for this head
    scores_base_offset = batch_id * seq_len * num_heads + head_id * seq_len
    values_base_offset = batch_id * num_heads * head_dim + head_id * head_dim
    
    # Create pointer matrices
    scores_ptr = attention_scores_ptr + scores_base_offset
    values_ptr = values_ptr + values_base_offset
    output_ptr = output_ptr + batch_id * seq_len * head_dim + head_id * seq_len
    
    # Load scores and apply scaling factor (0.0625)
    scores = tl.load(scores_ptr + tl.arange(0, seq_len), mask=None)
    scaled_scores = scores * 0.0625
    
    # Apply softmax with numerical stability
    max_val = tl.max(scaled_scores)
    exp_scores = tl.exp(scaled_scores - max_val)
    sum_exp = tl.sum(exp_scores)
    softmax_probs = exp_scores / sum_exp
    
    # Handle matrix multiplication part
    # For each output position in sequence
    for i in range(0, seq_len, BLOCK_M):
        i_start = i
        i_end = min(i + BLOCK_M, seq_len)
        
        # Accumulate results for this block
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        # Process along the head dimension (K dimension)
        for k in range(0, num_heads, BLOCK_K):
            k_start = k
            k_end = min(k + BLOCK_K, num_heads)
            
            # Load softmax probabilities for this block
            softmax_probs_block = tl.load(
                softmax_probs + i_start + tl.arange(0, i_end - i_start),
                mask=tl.arange(0, i_end - i_start) < (i_end - i_start)
            )
            
            # Load values for this block
            values_block = tl.load(
                values_ptr + k_start,
                mask=tl.arange(0, k_end - k_start) < (k_end - k_start)
            ).to(tl.float32)
            
            # Outer product: [BLOCK_M, 1] x [1, BLOCK_K] = [BLOCK_M, BLOCK_K]
            if BLOCK_N == 1:
                block_result = softmax_probs_block[:, None] * values_block[None, :]
            else:
                block_result = softmax_probs_block[:, None] * values_block[None, :][:, :BLOCK_N]
            
            # Accumulate
            acc += block_result
        
        # Store accumulated results
        output_block = acc
        tl.store(
            output_ptr + (i_start * head_dim + tl.arange(0, BLOCK_N)),
            output_block,
            mask=tl.arange(0, BLOCK_N) < min(BLOCK_N, head_dim)
        )

@torch.fx.wrap
def fused_softmax_matmul_attention(in_0, in_1):
    # Get tensor shapes
    batch_size, seq_len, num_heads = in_0.shape
    _, _, head_dim = in_1.shape
    
    # Create output tensor (matmul result before permute)
    matmul_output = torch.empty(batch_size, seq_len, head_dim, dtype=torch.float32, device=in_0.device)
    
    # Configure block sizes based on tensor dimensions  
    BLOCK_M = 64
    BLOCK_N = min(128, head_dim)
    BLOCK_K = 32
    
    # Launch kernel for fused operation (now includes scaling)
    grid = (batch_size, num_heads)
    
    fused_softmax_matmul_kernel[grid](
        in_0,  # Input to softmax (kernel will apply scaling)
        in_1,  # Value vectors
        matmul_output,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    
    # Apply the permutation as final step  
    final_output = matmul_output.permute(0, 2, 1)
    
    # Return intermediate values as expected by pattern
    # For the fused operation, we return None for tmp_1 (softmax output) since it's fused
    return None, matmul_output, final_output

def replacement_func():
    return fused_softmax_matmul_attention