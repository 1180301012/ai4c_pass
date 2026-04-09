import torch
import triton
import triton.language as tl

# Pattern matching function for layer normalization
def pattern(tmp_2, in_1, in_0):
    return torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)

# Argument extraction function  
def replacement_args(tmp_2, in_1, in_0):
    return (tmp_2, in_1, in_0)

# Optimized layer norm kernel
@triton.jit
def layernorm_kernel(
    input_ptr, 
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles one element of one sequence
    offsets = tl.arange(0, hidden_dim)
    pid = tl.program_id(0)
    
    # Calculate which sequence this program handles
    total_seqs = batch_size * seq_len
    if pid >= total_seqs:
        return  # Out of bounds
    
    # Calculate sequence and batch indices
    seq_idx = pid % seq_len
    batch_idx = pid // seq_len
    
    # Calculate starting offset for this sequence
    seq_offset = pid * hidden_dim
    
    # Load input slice for this sequence
    input_slice = tl.load(input_ptr + seq_offset + offsets, mask=offsets < hidden_dim, other=0.0)
    
    # Load weight and bias
    w = tl.load(weight_ptr + offsets, mask=offsets < hidden_dim, other=1.0)
    b = tl.load(bias_ptr + offsets, mask=offsets < hidden_dim, other=0.0)
    
    # Compute mean and variance with better precision
    mask_val = offsets < hidden_dim
    sum_val = tl.sum(input_slice)
    count = tl.sum(mask_val)
    local_mean = sum_val / (count + 1e-10)
    
    centered = input_slice - local_mean
    sum_sq = tl.sum(centered * centered)
    local_var = sum_sq / (count + 1e-10)
    
    # Apply normalization with better numerical stability
    std = tl.sqrt(local_var + eps)
    normalized = centered / std
    
    # Apply scale and shift
    out = normalized * w + b
    
    # Store result
    tl.store(output_ptr + seq_offset + offsets, out, mask=mask_val)

@torch.fx.wrap
def optimized_layernorm(tmp_2, in_1, in_0):
    batch_size, seq_len, hidden_dim = tmp_2.shape
    
    # Allocate output tensor
    output = torch.empty_like(tmp_2)
    eps = 1e-05
    
    # For small tensors, use single program to minimize overhead
    if batch_size * seq_len * hidden_dim <= 2048:
        # Single program handles all elements
        layernorm_kernel[(1,)](
            input_ptr=tmp_2,
            weight_ptr=in_1,
            bias_ptr=in_0,
            output_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            eps=eps,
        )
    else:
        # Larger tensors - one program per sequence
        num_programs = batch_size * seq_len
        layernorm_kernel[(num_programs,)](
            input_ptr=tmp_2,
            weight_ptr=in_1,
            bias_ptr=in_0,
            output_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            eps=eps,
        )
    
    return output

# Replacement function
def replacement_func():
    return optimized_layernorm