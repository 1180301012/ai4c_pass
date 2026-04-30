import torch
import triton
import triton.language as tl


# Triton-based linear (matmul + bias) kernel
@triton.jit
def triton_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,  # M=batch*seq, N=output_dim, K=input_dim
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel for linear: output[m,n] = sum_k(input[m,k] * weight[n,k]) + bias[n]
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program handles BLOCK_SIZE_M x BLOCK_SIZE_N block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    # Initialize accumulator with bias
    acc = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        mask_k = k_offs < K
        
        # Load input block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        input_block = tl.load(
            input_ptr + offs_m[:, None] * stride_input_m + k_offs[None, :] * stride_input_k,
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        ).to(tl.float32)
        
        # Load weight block [BLOCK_SIZE_N, BLOCK_SIZE_K] 
        weight_block = tl.load(
            weight_ptr + offs_n[:, None] * stride_weight_n + k_offs[None, :] * stride_weight_k,
            mask=(mask_n[:, None] & mask_k[None, :]),
            other=0.0
        ).to(tl.float32)
        
        # Compute matrix multiplication
        acc += tl.sum(input_block * weight_block, axis=1)
    
    # Store result
    tl.store(
        output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n,
        acc.to(tl.load(output_ptr).dtype),
        mask=(mask_m[:, None] & mask_n[None, :])
    )


@torch.fx.wrap
def triton_linear(input, weight, bias):
    """
    Triton-based linear operation.
    input: [batch, seq, K]
    weight: [N, K]
    bias: [N]
    output: [batch, seq, N]
    """
    batch_size, seq_len, K = input.shape
    N = weight.shape[0]
    M = batch_size * seq_len
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, N), dtype=input.dtype, device=input.device)
    
    # Strides for input [batch, seq, K]
    stride_input_m = seq_len * K  # stride for flattened M dimension
    stride_input_k = 1
    
    # Strides for weight [N, K]
    stride_weight_n = K
    stride_weight_k = 1
    
    # Strides for output [batch, seq, N]
    stride_output_m = seq_len * N
    stride_output_n = 1
    
    # Block sizes
    BLOCK_SIZE_M = 1  # Process 1 row (batch*seq element) per program in M dimension
    BLOCK_SIZE_N = 64  # Process 64 output elements per program
    BLOCK_SIZE_K = 64  # Reduction dimension
    
    # Grid: M dimension x N dimension
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    triton_linear_kernel[(grid_m, grid_n)](
        input, weight, bias,
        M, N, K,
        stride_input_m, stride_input_k,
        stride_weight_n, stride_weight_k,
        stride_output_m, stride_output_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output


@triton.jit
def fused_reshape_split_permute_kernel(
    input_ptr, output_q_ptr, output_k_ptr, output_v_ptr,
    batch_size, seq_len, num_heads, head_dim_qk, head_dim_v, hidden_dim,
    stride_in_b, stride_in_s,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    BLOCK_SIZE_QK: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr
):
    """
    Fused kernel for reshape + split + permute of QKV tensors.
    
    Input: [batch, seq, 1536] - already has linear applied
    Output Q: [batch, heads, seq, 32]
    Output K: [batch, heads, seq, 32]  
    Output V: [batch, heads, seq, 128]
    
    The input [batch, seq, 1536] is conceptually [batch, seq, 8, 192] split as:
    - Q: first 8*32=256 values -> permuted to [batch, 8, seq, 32]
    - K: next 8*32=256 values -> permuted to [batch, 8, seq, 32]
    - V: last 8*128=1024 values -> permuted to [batch, 8, seq, 128]
    """
    program_id = tl.program_id(0)
    num_elements = batch_size * seq_len
    
    if program_id >= num_elements:
        return
    
    batch_idx = program_id // seq_len
    seq_idx = program_id % seq_len
    
    # Compute input offset [batch, seq, 1536]
    input_offset = batch_idx * stride_in_b + seq_idx * stride_in_s
    
    # Process Q heads (8 heads, each 32 values)
    for q_head in range(8):
        head_offset = q_head * head_dim_qk
        q_vals = tl.load(input_ptr + input_offset + head_offset + tl.arange(0, BLOCK_SIZE_QK),
                         mask=tl.arange(0, BLOCK_SIZE_QK) < head_dim_qk, other=0.0)
        q_out_offset = batch_idx * stride_q_b + q_head * stride_q_h + seq_idx * stride_q_s
        tl.store(output_q_ptr + q_out_offset + tl.arange(0, BLOCK_SIZE_QK), q_vals,
                 mask=tl.arange(0, BLOCK_SIZE_QK) < head_dim_qk)
    
    # Process K heads (8 heads, each 32 values, offset by 256)
    k_base_offset = 256
    for k_head in range(8):
        head_offset = k_base_offset + k_head * head_dim_qk
        k_vals = tl.load(input_ptr + input_offset + head_offset + tl.arange(0, BLOCK_SIZE_QK),
                         mask=tl.arange(0, BLOCK_SIZE_QK) < head_dim_qk, other=0.0)
        k_out_offset = batch_idx * stride_k_b + k_head * stride_k_h + seq_idx * stride_k_s
        tl.store(output_k_ptr + k_out_offset + tl.arange(0, BLOCK_SIZE_QK), k_vals,
                 mask=tl.arange(0, BLOCK_SIZE_QK) < head_dim_qk)
    
    # Process V heads (8 heads, each 128 values, offset by 512)
    v_base_offset = 512
    for v_head in range(8):
        head_offset = v_base_offset + v_head * head_dim_v
        v_vals = tl.load(input_ptr + input_offset + head_offset + tl.arange(0, BLOCK_SIZE_V),
                         mask=tl.arange(0, BLOCK_SIZE_V) < head_dim_v, other=0.0)
        v_out_offset = batch_idx * stride_v_b + v_head * stride_v_h + seq_idx * stride_v_s
        tl.store(output_v_ptr + v_out_offset + tl.arange(0, BLOCK_SIZE_V), v_vals,
                 mask=tl.arange(0, BLOCK_SIZE_V) < head_dim_v)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the QKV projection pattern:
    linear(in_3, in_2, in_1) -> reshape -> split -> permute (3x) + transpose
    """
    linear_out = torch.nn.functional.linear(in_3, in_2, in_1)
    
    # Get batch size from input shape
    batch_size = in_3.shape[0]
    
    tmp_4 = linear_out.reshape(batch_size, 49, 8, -1)
    split = tmp_4.split([32, 32, 128], dim=3)
    
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    
    tmp_12 = in_0.to(device(type='cuda', index=0))
    tmp_13 = tmp_10.transpose(-2, -1)
    
    return tmp_9, tmp_12, tmp_13, tmp_11


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_qkv_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused QKV wrapper that combines:
    - Linear transformation (Triton-optimized)
    - Reshape to [batch, seq, heads, head_dim]
    - Split into Q, K, V
    - Permute to [batch, heads, seq, head_dim]
    - Transpose for K to [batch, heads, head_dim, seq]
    """
    batch_size, seq_len, hidden_dim = in_3.shape
    num_heads = 8
    head_dim_qk = 32
    head_dim_v = 128
    
    # Step 1: Linear projection using Triton kernel
    linear_out = triton_linear(in_3, in_2, in_1)
    
    # Output shapes after permute [batch, heads, seq, head_dim]
    q_shape = (batch_size, num_heads, seq_len, head_dim_qk)
    k_shape = (batch_size, num_heads, seq_len, head_dim_qk)
    v_shape = (batch_size, num_heads, seq_len, head_dim_v)
    
    # Allocate output tensors
    output_q = torch.empty(q_shape, dtype=in_3.dtype, device=in_3.device)
    output_k = torch.empty(k_shape, dtype=in_3.dtype, device=in_3.device)
    output_v = torch.empty(v_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Compute strides for linear output [batch, seq, 1536]
    stride_in_b = seq_len * 1536
    stride_in_s = 1536
    
    # Compute strides for output Q [batch, heads, seq, head_dim]
    stride_q_b = num_heads * seq_len * head_dim_qk
    stride_q_h = seq_len * head_dim_qk
    stride_q_s = head_dim_qk
    
    # Compute strides for output K [batch, heads, seq, head_dim]
    stride_k_b = num_heads * seq_len * head_dim_qk
    stride_k_h = seq_len * head_dim_qk
    stride_k_s = head_dim_qk
    
    # Compute strides for output V [batch, heads, seq, head_dim]
    stride_v_b = num_heads * seq_len * head_dim_v
    stride_v_h = seq_len * head_dim_v
    stride_v_s = head_dim_v
    
    num_elements = batch_size * seq_len
    BLOCK_SIZE = 128
    grid = (num_elements,)
    
    fused_reshape_split_permute_kernel[grid](
        linear_out, output_q, output_k, output_v,
        batch_size, seq_len, num_heads, head_dim_qk, head_dim_v,
        stride_in_b, stride_in_s,
        stride_q_b, stride_q_h, stride_q_s,
        stride_k_b, stride_k_h, stride_k_s,
        stride_v_b, stride_v_h, stride_v_s,
        BLOCK_SIZE
    )
    
    # Move in_0 to cuda
    tmp_12 = in_0.to(device='cuda')
    
    # Transpose K: [batch, heads, seq, head_dim] -> [batch, heads, head_dim, seq]
    tmp_13 = output_k.transpose(-2, -1)
    
    return output_q, tmp_12, tmp_13, output_v


def replacement_func():
    return fused_qkv_wrapper