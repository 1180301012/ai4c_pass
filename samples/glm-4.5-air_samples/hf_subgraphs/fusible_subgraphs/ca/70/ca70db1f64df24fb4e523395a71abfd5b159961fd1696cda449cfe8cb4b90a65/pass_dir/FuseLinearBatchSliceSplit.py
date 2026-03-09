import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, other_weight, other_bias):
    # Pattern: batched linear transformation followed by splitting last dimension
    tmp_10 = torch.nn.functional.linear(x, weight, bias)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12

def replacement_args(x, weight, bias, other_weight, other_bias):
    return (x, weight, bias)

@triton.jit
def batched_linear_split_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out1_ptr, out2_ptr,
    n_batch, n_seq, n_in_cols,
    n_out_cols_half,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate indices
    batch_idx = pid_m
    seq_idx = pid_n
    
    row_start = batch_idx * BLOCK_SIZE_M + seq_idx * BLOCK_SIZE_M * n_seq
    seq_start = seq_idx * BLOCK_SIZE_N
    seq_end = min(seq_start + BLOCK_SIZE_N, n_seq)
    
    if batch_idx >= n_batch or seq_idx >= (tl.cdiv(n_seq, BLOCK_SIZE_N)):
        return
    
    # Load bias for this output segment
    bias_segment = tl.load(bias_ptr)
    
    # Process output positions
    for k in range(seq_start, seq_end):
        # Initialize accumulators
        acc1 = tl.zeros((n_out_cols_half,), dtype=tl.float32)
        acc2 = tl.zeros((n_out_cols_half,), dtype=tl.float32)
        
        # Matrix multiplication for input sequence
        for j in range(n_in_cols):
            # Load input element
            x_val = tl.load(x_ptr + batch_idx * n_seq * n_in_cols + seq_idx * n_in_cols + j)
            
            # Load weight segments
            weights1 = tl.load(weight_ptr + j * n_in_cols + k * 2 * n_out_cols_half)
            weights2 = tl.load(weight_ptr + j * n_in_cols + (k * 2 + 1) * n_out_cols_half)
            
            # Vectorized accumulation
            for i in range(n_out_cols_half):
                acc1[i] += x_val * weights1
                acc2[i] += x_val * weights2
        
        # Add bias
        for i in range(n_out_cols_half):
            acc1[i] += bias_segment[i]
            acc2[i] += bias_segment[i + n_out_cols_half]
        
        # Store results
        out_offset = batch_idx * n_seq * 2 * n_out_cols_half + seq_idx * 2 * n_out_cols_half + k * 2 * n_out_cols_half
        tl.store(out1_ptr + out_offset + k * n_out_cols_half, acc1)
        tl.store(out2_ptr + out_offset + k * n_out_cols_half, acc2)

@torch.fx.wrap
def fused_batched_linear_split(x, weight, bias):
    n_batch, n_seq, n_in_cols = x.shape
    n_out_cols = weight.shape[0]
    n_out_cols_half = n_out_cols // 2
    
    out1 = torch.empty((n_batch, n_seq, n_out_cols_half), dtype=x.dtype, device=x.device)
    out2 = torch.empty((n_batch, n_seq, n_out_cols_half), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 1  # Process one batch at a time
    BLOCK_SIZE_N = 32  # Process multiple sequence positions
    
    grid = (
        triton.cdiv(n_batch, BLOCK_SIZE_M),
        triton.cdiv(n_seq, BLOCK_SIZE_N)
    )
    
    batched_linear_split_kernel[grid](
        x, weight, bias,
        out1, out2,
        n_batch, n_seq, n_in_cols,
        n_out_cols_half,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out1, out2

def replacement_func():
    return fused_batched_linear_split