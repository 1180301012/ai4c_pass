import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_2, in_1):
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6

def replacement_args(tmp_5, in_2, in_1):
    return (tmp_5, in_2, in_1)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_K: tl.constexpr,
    eps: tl.constexpr,
):
    # For each vector to normalize (each row's last dim)
    program_id = tl.program_id(0)
    vec_idx = program_id  # Each program handles one vector of size n_cols
    
    if vec_idx >= n_rows:
        return  # Don't process if out of bounds
    
    # Load the entire vector (for n_cols <= 384, this is reasonable)
    offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = offsets < n_cols
    
    # Load vector data
    vec_data = tl.load(x_ptr + vec_idx * n_cols + offsets, mask=mask)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # Compute mean and variance
    mean = tl.sum(vec_data) / n_cols
    var = tl.sum((vec_data - mean) * (vec_data - mean)) / n_cols
    std = tl.sqrt(var + eps)
    
    # Normalize: y = (x - mean) / std * weight + bias
    normalized = (vec_data - mean) / std
    out_vec = normalized * weight + bias
    
    # Store result
    out_ptrs = out_ptr + vec_idx * n_cols + offsets
    tl.store(out_ptrs, out_vec, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias):
    # Always handle as 2D for simplicity - flatten if needed
    if len(x.shape) == 3:
        batch_size, seq_len, hidden_dim = x.shape
        # Calculate flattened size
        total_vectors = batch_size * seq_len
        out = torch.empty_like(x)
        
        # Use a block size larger than 384 to ensure we cover all columns
        BLOCK_SIZE_K = 512
        
        # Launch kernel for flattened tensor by treating memory as 2D
        layernorm_kernel[(total_vectors,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            n_rows=total_vectors,
            n_cols=hidden_dim,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            eps=1e-12
        )
    else:
        # 2D tensor
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        
        # Use a block size larger than n_cols to ensure we cover all columns
        BLOCK_SIZE_K = max(n_cols, 512)
        
        layernorm_kernel[(n_rows,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            eps=1e-12
        )
    
    return out

def replacement_func():
    return optimized_layernorm