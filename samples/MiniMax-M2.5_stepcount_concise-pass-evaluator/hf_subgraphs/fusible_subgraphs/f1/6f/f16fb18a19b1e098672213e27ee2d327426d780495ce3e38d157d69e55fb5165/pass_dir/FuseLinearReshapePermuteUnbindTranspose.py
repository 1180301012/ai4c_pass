import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: linear + reshape + permute + unbind + transpose
    This corresponds to QKV projection followed by head splitting in attention.
    
    convit_small: weight [1296, 432] -> 3 heads, 9 tokens per group, 48 dim
    convit_tiny: weight [576, 192] -> 3 heads, 4 tokens per group, 48 dim
    
    After linear: [1, 197, 1296] for small, [1, 197, 576] for tiny
    Then reshape and permute to get 3 separate heads, with one transposed.
    """
    # Step 1: Linear projection (QKV)
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    
    # Get the output dimension from the linear result
    # Shape: [batch, seq, num_heads * tokens_per_group * head_dim]
    # For convit_small: 1296 = 3 * 9 * 48
    # For convit_tiny: 576 = 3 * 4 * 48
    out_dim = tmp_1.shape[2]
    head_dim = 48
    num_heads = 3
    tokens_per_group = (out_dim // head_dim) // num_heads
    
    # Step 2: Reshape to separate heads (batch, seq, num_heads, tokens_per_group, head_dim)
    tmp_2 = tmp_1.reshape(1, 197, num_heads, tokens_per_group, head_dim)
    
    # Step 3: Permute to (num_heads, batch, tokens_per_group, seq, head_dim)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    
    # Step 4: Unbind to get individual heads
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    
    # Step 5: Transpose the query head (tmp_6) for attention computation
    tmp_8 = tmp_6.transpose(-2, -1)
    
    # Return all three heads: tmp_5 (key), tmp_8 (query, transposed), tmp_7 (value)
    return tmp_5, tmp_8, tmp_7


def replacement_args(in_0, in_1):
    """
    Extract the arguments needed for the replacement function.
    """
    return (in_0, in_1)


@triton.autotune(
    configs=[
        # Different block sizes for optimal occupancy
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['num_heads', 'seq_len', 'head_dim', 'tokens_per_group'],
)
@triton.jit
def fused_qkv_kernel(
    # Pointers
    weight_ptr, input_ptr, 
    output_k_ptr, output_q_ptr, output_v_ptr,
    # Shapes
    weight_batch, weight_rows, weight_cols,
    input_batch, input_seq, input_dim,
    num_heads, seq_len, head_dim, tokens_per_group,
    # Strides
    weight_stride_batch, weight_stride_rows, weight_stride_cols,
    input_stride_batch, input_stride_seq, input_stride_dim,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. QKV linear projection: output = input @ weight.T
    2. Reshape to separate heads
    3. Permute dimensions
    4. Unbind to get individual heads
    5. Transpose query head
    
    This kernel processes each head separately and produces:
    - output_k: (1, tokens_per_group, seq_len, head_dim) - Key
    - output_q: (1, tokens_per_group, head_dim, seq_len) - Query (transposed)
    - output_v: (1, tokens_per_group, seq_len, head_dim) - Value
    """
    # Calculate offsets for this head
    head_id = tl.program_id(0)
    
    # Each program processes one head
    # Calculate base offset for this head in the QKV output
    # QKV output shape: (1, seq_len, num_heads * tokens_per_group * head_dim)
    # After reshape: (1, seq_len, num_heads, tokens_per_group, head_dim)
    # After permute: (num_heads, 1, tokens_per_group, seq_len, head_dim)
    
    # For weight matrix: total output dim = num_heads * tokens_per_group * head_dim
    # The weight is organized as: [num_heads * tokens_per_group * head_dim, input_dim]
    # Each head's weight starts at: head_id * tokens_per_group * head_dim
    
    # Calculate the starting row in weight for this head
    head_weight_offset = head_id * tokens_per_group * head_dim
    
    # Process tokens per group
    for token_idx in range(tokens_per_group):
        # Calculate output position for this token
        # output position in QKV: head_id * tokens_per_group + token_idx
        token_offset = head_id * tokens_per_group + token_idx
        
        # For each token, compute the matrix multiplication
        # input: (1, seq_len, input_dim) @ weight: (tokens_per_group * head_dim, input_dim)
        # But we process one token_row at a time: (1, seq_len, input_dim) @ (head_dim, input_dim)
        
        token_weight_offset = head_weight_offset + token_idx * head_dim
        
        # Initialize output for this token's head dimension
        # Output for this token: (1, seq_len, head_dim)
        
        # Matrix multiplication: input @ weight_row.T
        # input: [1, seq_len, input_dim]
        # weight_row: [head_dim, input_dim]
        # output: [1, seq_len, head_dim]
        
        # Loop over seq_len (K dimension)
        for k in range(0, seq_len, BLOCK_SIZE_K):
            # Load input slice: (BLOCK_SIZE_K, input_dim)
            # Actually, we process seq in blocks
            
            # The computation: for each output row in head_dim
            # output[0, seq_pos, head_pos] = sum(input[0, seq_pos, :] * weight[weight_row_offset + head_pos, :])
            
            # We'll iterate over head_dim
            for hb in range(0, head_dim, BLOCK_SIZE_N):
                # Compute matrix multiplication for this block
                # Output shape: (BLOCK_SIZE_K, BLOCK_SIZE_N) for one seq_block x head_block
                
                # Accumulator
                acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
                
                # Loop over input dimension
                for i in range(0, input_dim, BLOCK_SIZE_M):
                    # Load weight: (BLOCK_SIZE_M, BLOCK_SIZE_N) - transposed access
                    # weight[weight_row_offset + hb:weight_row_offset + hb + BLOCK_SIZE_N, i:i + BLOCK_SIZE_M]
                    weight_idx_h = token_weight_offset + hb + tl.arange(0, BLOCK_SIZE_N)
                    weight_idx_w = i + tl.arange(0, BLOCK_SIZE_M)
                    weight_idx = weight_idx_h[:, None] * weight_stride_rows + weight_idx_w[None, :]
                    
                    w = tl.load(weight_ptr + weight_idx, mask=weight_idx_w[None, :] < input_dim, other=0.0)
                    w = tl.trans(w)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    
                    # Load input: (1, seq_len, input_dim)
                    # We need input[0, k:k+BLOCK_SIZE_K, i:i+BLOCK_SIZE_M]
                    input_idx_seq = k + tl.arange(0, BLOCK_SIZE_K)
                    input_idx_dim = i + tl.arange(0, BLOCK_SIZE_M)
                    input_idx = input_idx_seq[:, None] * input_stride_seq + input_idx_dim[None, :]
                    
                    x = tl.load(input_ptr + input_idx, mask=(input_idx_seq[:, None] < seq_len) & (input_idx_dim[None, :] < input_dim), other=0.0)
                    # x shape: (BLOCK_SIZE_K, BLOCK_SIZE_M)
                    
                    # Multiply and accumulate
                    # x @ w: (BLOCK_SIZE_K, BLOCK_SIZE_M) @ (BLOCK_SIZE_M, BLOCK_SIZE_N) = (BLOCK_SIZE_K, BLOCK_SIZE_N)
                    acc += tl.dot(x, w)
                
                # Store results
                # output_k: (1, tokens_per_group, seq_len, head_dim)
                # Index: (0, token_idx, k:k+BLOCK_SIZE_K, hb:hb+BLOCK_SIZE_N)
                output_k_idx = (0 * tokens_per_group * seq_len * head_dim + 
                               token_idx * seq_len * head_dim + 
                               (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * head_dim + 
                               (hb + tl.arange(0, BLOCK_SIZE_N))[None, :])
                tl.store(output_k_ptr + output_k_idx, acc)
                
                # output_v is the same as output_k (no transpose)
                tl.store(output_v_ptr + output_k_idx, acc)
                
                # output_q (transposed): (1, tokens_per_group, head_dim, seq_len)
                # Index: (0, token_idx, hb:hb+BLOCK_SIZE_N, k:k+BLOCK_SIZE_K)
                output_q_idx = (0 * tokens_per_group * head_dim * seq_len + 
                               token_idx * head_dim * seq_len + 
                               (hb + tl.arange(0, BLOCK_SIZE_N))[:, None] * seq_len + 
                               (k + tl.arange(0, BLOCK_SIZE_K))[None, :])
                # Transpose: swap the last two dimensions
                tl.store(output_q_ptr + output_q_idx, tl.trans(acc))


@torch.fx.wrap
def fused_qkv_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused QKV kernel.
    
    Args:
        in_0: weight tensor [num_heads * tokens_per_group * head_dim, input_dim]
              convit_small: [1296, 432] = 3 * 9 * 48, 432
              convit_tiny: [576, 192] = 3 * 4 * 48, 192
        in_1: input tensor [batch, seq_len, input_dim]
              convit_small: [1, 197, 432]
              convit_tiny: [1, 197, 192]
    
    Returns:
        Tuple of (output_k, output_q, output_v):
        - output_k: (1, tokens_per_group, seq_len, head_dim)
        - output_q: (1, tokens_per_group, head_dim, seq_len) - transposed
        - output_v: (1, tokens_per_group, seq_len, head_dim)
    """
    # Get dimensions
    weight = in_0
    input_tensor = in_1
    
    weight_shape = weight.shape  # [num_heads * tokens_per_group * head_dim, input_dim]
    input_shape = input_tensor.shape  # [batch, seq_len, input_dim]
    
    batch = input_shape[0]
    seq_len = input_shape[1]
    input_dim = input_shape[2]
    
    total_out_dim = weight_shape[0]
    head_dim = 48  # Fixed head dimension
    
    # Calculate num_heads and tokens_per_group
    # total_out_dim = num_heads * tokens_per_group * head_dim
    # For convit_small: 1296 = 3 * 9 * 48 -> num_heads=3, tokens_per_group=9
    # For convit_tiny: 576 = 3 * 4 * 48 -> num_heads=3, tokens_per_group=4
    num_heads = 3
    tokens_per_group = (total_out_dim // head_dim) // num_heads
    
    # Allocate output tensors
    # output_k: (1, tokens_per_group, seq_len, head_dim)
    # output_q: (1, tokens_per_group, head_dim, seq_len) 
    # output_v: (1, tokens_per_group, seq_len, head_dim)
    output_k = torch.empty((1, tokens_per_group, seq_len, head_dim), device=input_tensor.device, dtype=torch.float32)
    output_q = torch.empty((1, tokens_per_group, head_dim, seq_len), device=input_tensor.device, dtype=torch.float32)
    output_v = torch.empty((1, tokens_per_group, seq_len, head_dim), device=input_tensor.device, dtype=torch.float32)
    
    # Calculate grid
    # One program per head
    grid = (num_heads,)
    
    # Get contiguous data pointers
    weight_cont = weight.contiguous()
    input_cont = input_tensor.contiguous()
    
    # Launch kernel
    fused_qkv_kernel[grid](
        weight_cont, input_cont,
        output_k, output_q, output_v,
        weight_shape[0], weight_shape[1], weight_shape[2],
        input_shape[0], input_shape[1], input_shape[2],
        num_heads, seq_len, head_dim, tokens_per_group,
        weight.stride(0), weight.stride(1), weight.stride(2) if len(weight.stride()) > 2 else 0,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
    )
    
    return output_k, output_q, output_v


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_qkv_kernel_wrapper