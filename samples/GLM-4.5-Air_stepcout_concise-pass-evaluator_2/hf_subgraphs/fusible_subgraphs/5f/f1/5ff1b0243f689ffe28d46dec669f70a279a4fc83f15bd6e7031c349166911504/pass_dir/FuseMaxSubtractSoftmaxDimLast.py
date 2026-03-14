import torch
import triton
import triton.language as tl

# Pattern matching function - matches the max-expand-subtract-softmax sequence
def pattern(in_0, in_1):
    # Match the exact computation from the model
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    # Return what the original computation returns (both softmax output and reshaped in_1)
    return tmp_4, in_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for softmax along last dimension
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    feature_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in batch and feature dimensions
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate the offset for this batch and feature
    offset = batch_idx * feature_size * seq_len + feature_idx * seq_len
    
    # Load the sequence for this batch and feature combination
    seq_ptr = x_ptr + offset
    seq_data = tl.load(seq_ptr, mask=tl.arange(0, seq_len) < seq_len, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(seq_data)
    
    # Subtract max and compute exp
    shifted = seq_data - max_val
    exp_vals = tl.exp(shifted)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_vals)
    
    # Compute softmax
    softmax_vals = exp_vals / sum_exp
    
    # Store the result
    out_ptr_ptr = out_ptr + offset
    tl.store(out_ptr_ptr, softmax_vals, mask=tl.arange(0, seq_len) < seq_len)

# Kernel wrapper 
@torch.fx.wrap
def fused_max_subtract_softmax(in_0):
    batch_size, feature_size, seq_len = in_0.shape
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel with proper grid configuration (each program handles one batch and feature)
    # The grid covers all batch and feature combinations
    softmax_kernel[(batch_size, feature_size)](
        in_0,
        out,
        batch_size,
        feature_size,
        seq_len,
        seq_len,  # BLOCK_SIZE is the sequence length for each softmax computation
    )
    
    return out

# Replacement function
def replacement_func():
    def optimized_forward(in_0, in_1):
        # Apply fused operation to in_0
        softmax_output = fused_max_subtract_softmax(in_0)
        return softmax_output, in_1
    
    return optimized_forward