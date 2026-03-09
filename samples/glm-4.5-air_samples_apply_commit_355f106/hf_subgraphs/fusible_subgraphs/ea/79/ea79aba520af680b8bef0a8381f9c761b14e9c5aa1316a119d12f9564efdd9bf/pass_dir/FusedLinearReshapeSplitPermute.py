import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate work per program - each program handles one output element
    row_idx = pid_m  # Flattened batch * seq index
    col_idx = pid_n  # Output feature index
    
    # Bounds checking
    row_mask = row_idx < (batch_size * seq_len)
    col_mask = col_idx < out_features
    
    if not row_mask or not col_mask:
        return
        
    # Calculate original coordinates
    batch_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len
    
    # Load bias for this output position
    bias_val = tl.load(bias_ptr + col_idx, mask=col_mask, other=0.0)
    acc = bias_val.to(tl.float32)
    
    # Process input features in chunks
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, in_features)
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < in_features
        
        # Load weight chunk for this output feature
        weight_chunk = tl.load(
            weight_ptr + col_idx * in_features + k_offset,
            mask=k_mask,
            other=0.0
        )
        
        # Load input chunk for this batch and sequence
        input_idx = (batch_idx * seq_len * in_features + seq_idx * in_features + k_offset)
        input_chunk = tl.load(
            x_ptr + input_idx,
            mask=k_mask,
            other=0.0
        )
        
        # Compute dot product using element-wise multiplication and sum
        acc += tl.sum(input_chunk.to(tl.float32) * weight_chunk.to(tl.float32))
    
    # Calculate output index and store
    output_idx = row_idx * out_features + col_idx
    tl.store(out_ptr + output_idx, acc, row_mask & col_mask)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    
    # Output shape
    out_shape = (batch_size, seq_len, out_features)
    output = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Thread block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    linear_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_linear