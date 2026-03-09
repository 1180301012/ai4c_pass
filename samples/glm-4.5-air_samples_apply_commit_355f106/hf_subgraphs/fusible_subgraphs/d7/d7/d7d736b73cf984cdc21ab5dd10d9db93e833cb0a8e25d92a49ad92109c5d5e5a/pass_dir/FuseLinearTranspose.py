import torch
import triton
import triton.language as tl
import math

def pattern(input, weight, bias):
    # Original computation: linear followed by transpose(-1, -2)
    tmp = torch.nn.functional.linear(input, weight, bias)
    result = tmp.transpose(-1, -2)
    return result

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def fused_linear_transpose_kernel(
    input_ptr,  # [batch, seq_len, in_features]
    weight_ptr,  # [out_features, in_features]
    bias_ptr,   # [out_features]
    output_ptr, # [batch, out_features, seq_len]
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,  # batch dimension tile size
    BLOCK_SIZE_N: tl.constexpr,  # out_features dimension tile size
    SEQ_LEN_64: tl.constexpr,    # sequence length rounded to power of 2
):
    # Each program handles a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute tile boundaries
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = min(m_start + BLOCK_SIZE_M, batch_size)
    n_end = min(n_start + BLOCK_SIZE_N, out_features)
    
    # Initialize output tile
    acc = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float32)
    
    # Loop over sequence positions
    for k in range(0, seq_len):
        # Load input for this batch and sequence position
        mask_m = (tl.arange(m_end - m_start) + m_start) < batch_size
        mask_n = (tl.arange(n_end - n_start) + n_start) < out_features
        
        # Load input vectors: batch x seq_len, extract one seq_len position
        input_ptr_batch = input_ptr + (m_start * seq_len * in_features + k * in_features)
        input_vals = tl.load(input_ptr_batch + tl.arange(0, in_features), 
                           mask=tl.arange(0, in_features) < in_features, other=0.0)
        
        # Load weight vectors: out_features x in_features
        weight_ptr_batch = weight_ptr + (n_start * in_features)
        weight_vals = tl.load(weight_ptr_batch + tl.arange(0, in_features), 
                            mask=tl.arange(0, in_features) < in_features, other=0.0)
        
        # Compute dot product for each batch/out_features pair
        for m in range(m_end - m_start):
            for n in range(n_end - n_start):
                if mask_m[m] and mask_n[n]:
                    acc[m, n] = acc[m, n] + tl.sum(input_vals * weight_vals)
    
    # Load bias and add for each output feature
    bias_vals = tl.load(bias_ptr + n_start, mask=n_start < out_features, other=0.0)
    for m in range(m_end - m_start):
        for n in range(n_end - n_start):
            if mask_m[m] and mask_n[n]:
                add_offset = (pid_n * BLOCK_SIZE_N + n) - n_start
                acc[m, n] = acc[m, n] + bias_vals[add_offset]
    
    # Store results for all sequence positions
    for k in range(seq_len):
        for m in range(m_end - m_start):
            for n in range(n_end - n_start):
                if mask_m[m] and mask_n[n]:
                    batch_idx = m_start + m
                    out_features_idx = n_start + n
                    output_offset = batch_idx * out_features * seq_len + out_features_idx * seq_len + k
                    tl.store(output_ptr + output_offset, acc[m, n])

@torch.fx.wrap
def fused_linear_transpose(input, weight, bias):
    # Get input dimensions
    batch_size, seq_len, in_features = input.shape
    out_features = weight.shape[0]
    
    # Initialize output
    output = torch.empty((batch_size, out_features, seq_len), 
                        dtype=input.dtype, device=input.device)
    
    # Optimized Triton kernel launch configuration with block-based tiling
    BLOCK_SIZE_M = 8   # Batch dimension tile for better occupancy
    BLOCK_SIZE_N = 64  # Out features dimension tile (matches weight matrix width)
    BLOCK_SIZE_K = 32  # Parallel reduction size for matrix mult
    
    # Calculate grid dimensions for tiled computation
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel with optimized tiling
    fused_linear_transpose_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch_size,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SEQ_LEN=seq_len
    )
    
    return output

def replacement_func():
    return fused_linear_transpose