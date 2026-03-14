import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 128, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 256, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 512, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 128, 'num_warps': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 256, 'num_warps': 8}, num_stages=3),
    ],
    key=['batch', 'seq_len'],
)
@triton.jit
def fused_all_ops_kernel(
    # Input pointers
    in_5_ptr, in_4_ptr, in_6_ptr, emb_ptr,
    in_0_ptr,
    # Layer norm params
    ln_weight_ptr, ln_bias_ptr,
    # Output pointers
    output_ptr, ln_output_ptr,
    # Strides
    s_in5_b, s_in5_s, s_in5_h,
    s_in4_b, s_in4_s, s_in4_h,
    s_in6_b, s_in6_s,
    s_emb_v, s_emb_h,
    s_in0_b, s_in0_s,
    s_out_b, s_out_s, s_out_h,
    # Dimensions
    batch, seq_len, hidden_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel for all operations: div, cast, embedding, add, unsqueeze, mul, cast, layer_norm"""
    
    # Get program coordinates
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute offsets
    batch_offset = pid_b * s_out_s * s_out_h
    seq_offset = pid_m * BLOCK_SIZE_M * s_out_h
    hidden_offset = pid_n * BLOCK_SIZE_N
    
    # Compute actual block sizes
    seq_remainder = seq_len - pid_m * BLOCK_SIZE_M
    hidden_remainder = hidden_dim - pid_n * BLOCK_SIZE_N
    block_m = tl.minimum(BLOCK_SIZE_M, seq_remainder)
    block_n = tl.minimum(BLOCK_SIZE_N, hidden_remainder)
    
    # Initialize layer norm accumulators
    sum_vals = tl.zeros((block_m, block_n), tl.float32)
    sum_sq_vals = tl.zeros((block_m, block_n), tl.float32)
    count = 0
    
    # Process each element
    for i in range(block_m):
        seq_idx = pid_m * BLOCK_SIZE_M + i
        
        # === Division and Cast: tmp_5 = (in_5 / in_4).to(float32) ===
        in5_offset = batch_offset + seq_idx * s_in5_h
        in4_offset = batch_offset + seq_idx * s_in4_h
        
        for j in range(block_n):
            h_idx = pid_n * BLOCK_SIZE_N + j
            
            # Load in_5 and in_4
            in5_val = tl.load(in_5_ptr + in5_offset + h_idx * s_in5_h).to(tl.float32)
            in4_val = tl.load(in_4_ptr + in4_offset + h_idx * s_in4_h).to(tl.float32)
            
            # Division
            div_result = in5_val / in4_val
            
            # === Embedding lookup: tmp_6 ===
            pos_id = tl.load(in_6_ptr + seq_idx * s_in6_s + pid_b * s_in6_b).to(tl.int64)
            emb_offset = pos_id * s_emb_h + h_idx
            emb_val = tl.load(emb_ptr + emb_offset).to(tl.float32)
            
            # === Addition: tmp_7 ===
            add_result = div_result + emb_val
            
            # === Unsqueeze and Multiply: tmp_9 ===
            # in_0 is [batch, seq], unsqueeze(-1) makes it [batch, seq, 1]
            # Multiplication broadcasts over hidden dim
            mask_val = tl.load(in_0_ptr + seq_idx * s_in0_s + pid_b * s_in0_b).to(tl.float32)
            mul_result = add_result * mask_val
            
            # Store result (this is tmp_10 after cast)
            tl.store(output_ptr + batch_offset + seq_idx * s_out_h + h_idx, mul_result)
            
            # Accumulate for layer norm
            sum_vals = sum_vals + mul_result
            sum_sq_vals = sum_sq_vals + mul_result * mul_result
            count += 1
    
    # === Layer Normalization ===
    # Compute mean and variance
    mean = sum_vals / tl.maximum(count, 1)
    variance = (sum_sq_vals / tl.maximum(count, 1)) - mean * mean
    std = tl.sqrt(variance + eps)
    
    # Apply layer norm
    for i in range(block_m):
        seq_idx = pid_m * BLOCK_SIZE_M + i
        for j in range(block_n):
            h_idx = pid_n * BLOCK_SIZE_N + j
            
            # Reload value
            val = tl.load(output_ptr + batch_offset + seq_idx * s_out_h + h_idx)
            
            # Normalize
            normalized = (val - mean) / std
            gamma = tl.load(ln_weight_ptr + h_idx)
            beta = tl.load(ln_bias_ptr + h_idx)
            ln_out = normalized * gamma + beta
            
            tl.store(ln_output_ptr + batch_offset + seq_idx * s_out_h + h_idx, ln_out)


@torch.fx.wrap
def fused_all_ops_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Fully fused Triton kernel that combines all operations.
    
    Args:
        in_0: attention_mask [batch, seq]
        in_1: embedding weight [vocab_size, hidden]
        in_2: layer norm bias [hidden]
        in_3: layer norm weight [hidden]
        in_4: divisor [batch, seq, 1] or [batch, 1, 1]
        in_5: numerator [batch, seq, hidden]
        in_6: position_ids [batch, seq]
    
    Returns:
        output: [batch, seq, hidden]
        layernorm_output: [batch, seq, hidden]
    """
    batch = in_5.shape[0]
    seq_len = in_5.shape[1]
    hidden_dim = 1280
    
    # Allocate outputs
    output = torch.empty_like(in_5, dtype=torch.float32, device=in_5.device)
    ln_output = torch.empty_like(in_5, dtype=torch.float32, device=in_5.device)
    
    # Compute grid
    # Use finer grid for better parallelism
    grid_b = batch
    grid_m = min(4, (seq_len + 3) // 4) if seq_len > 4 else 1
    grid_n = (hidden_dim + 127) // 128
    
    grid = (grid_b, grid_m, grid_n)
    
    # Get strides
    s_in5 = in_5.stride()
    s_in4 = in_4.stride()
    s_in6 = in_6.stride()
    s_emb = in_1.stride()
    s_in0 = in_0.stride()
    s_out = output.stride()
    
    # Launch kernel
    fused_all_ops_kernel[grid](
        in_5, in_4, in_6, in_1,
        in_0,
        in_3, in_2,
        output, ln_output,
        s_in5[0], s_in5[1], s_in5[2] if in_5.dim() > 2 else 0,
        s_in4[0], s_in4[1], s_in4[2] if in_4.dim() > 2 else 0,
        s_in6[0], s_in6[1],
        s_emb[0], s_emb[1],
        s_in0[0], s_in0[1],
        s_out[0], s_out[1], s_out[2] if output.dim() > 2 else 0,
        batch, seq_len, hidden_dim,
        1e-12,
    )
    
    return output, ln_output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the pattern from the model:
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_6 = torch.nn.functional.embedding(in_6, tmp_1, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = tmp_0.unsqueeze(-1)
    tmp_9 = tmp_7 * tmp_8
    tmp_10 = tmp_9.to(torch.float32)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1280,), tmp_3, tmp_2, 1e-12)
    return (tmp_10, tmp_11)
    """
    # Identity assignments
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    
    # Division
    tmp_4 = in_5 / in_4
    # Cast to float32
    tmp_5 = tmp_4.to(torch.float32)
    # Embedding lookup 
    tmp_6 = torch.nn.functional.embedding(in_6, tmp_1, 1, None, 2.0, False, False)
    # Addition
    tmp_7 = tmp_5 + tmp_6
    # Unsqueeze
    tmp_8 = tmp_0.unsqueeze(-1)
    # Multiplication
    tmp_9 = tmp_7 * tmp_8
    # Cast to float32
    tmp_10 = tmp_9.to(torch.float32)
    # Layer norm
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1280,), tmp_3, tmp_2, 1e-12)
    return (tmp_10, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Extract arguments for the replacement function.
    For this pass, we pass through all arguments.
    The kernel will compute:
    1. tmp_5 = (in_5 / in_4).to(float32)
    2. tmp_6 = embedding(in_6, in_1)
    3. tmp_7 = tmp_5 + tmp_6
    4. tmp_8 = in_0.unsqueeze(-1)
    5. tmp_9 = tmp_7 * tmp_8
    6. tmp_10 = tmp_9.to(float32)
    7. tmp_11 = layer_norm(tmp_10, in_3, in_2)
    
    Returns: (all inputs needed for the fused computation)
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    """
    Returns the fused kernel function.
    """
    return fused_all_ops_wrapper