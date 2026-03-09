import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr,  # hidden_states: [batch, seq_len, hidden_dim]
    weight_ptr,  # weight: [out_features, hidden_dim] 
    out_ptr,  # output: [batch, seq_len, out_features]
    batch, seq_len, hidden_dim, out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program processes one output element
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // (seq_len * out_features)
    seq_idx = (pid // out_features) % seq_len  
    feat_idx = pid % out_features
    
    # Bounds check - use separate nested checks to avoid chained operators
    if batch_idx >= batch:
        return
    if seq_idx >= seq_len:
        return
    if feat_idx >= out_features:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over hidden dimension for matrix multiplication
    # result[b, s, f] = sum(h hidden_states[b, s, h] * weight[f, h])
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # Load input block for current batch, sequence position
        input_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + k
        input_ptr = x_ptr + input_offset
        input_block = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE_K), 
                            mask=k + tl.arange(0, BLOCK_SIZE_K) < hidden_dim, 
                            other=0.0)
        
        # Load weight block for current feature
        weight_offset = feat_idx * hidden_dim + k
        weight_ptr_base = weight_ptr + weight_offset
        weight_block = tl.load(weight_ptr_base + tl.arange(0, BLOCK_SIZE_K), 
                             mask=k + tl.arange(0, BLOCK_SIZE_K) < hidden_dim, 
                             other=0.0)
        
        # Dot product and accumulate
        acc += tl.sum(input_block * weight_block)
    
    # Store result: [batch_idx, seq_idx, feat_idx]
    output_offset = batch_idx * seq_len * out_features + seq_idx * out_features + feat_idx
    tl.store(out_ptr + output_offset, acc)

@torch.fx.wrap  
def optimized_linear(hidden_states, weight):
    """
    Optimized linear operation using Triton
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    out_features, _ = weight.shape
    
    # Output shape same as original linear: [batch, seq_len, out_features]
    output = torch.empty((batch, seq_len, out_features), 
                        dtype=hidden_states.dtype, 
                        device=hidden_states.device)
    
    # Calculate grid size
    total_elements = batch * seq_len * out_features
    BLOCK_SIZE = 256
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    linear_kernel[(grid_size,)](
        hidden_states,
        weight, 
        output,
        batch, seq_len, hidden_dim, out_features,
        BLOCK_SIZE, 64, 32
    )
    
    return output

def pattern(hidden_states, weight):
    """
    Pattern: torch.nn.functional.linear operation
    """
    return torch.nn.functional.linear(hidden_states, weight, None)

def replacement_args(hidden_states, weight):
    """
    Extract arguments needed for the replacement function
    """
    return (hidden_states, weight)

def replacement_func():
    """
    Return the optimized kernel function
    """
    return optimized_linear