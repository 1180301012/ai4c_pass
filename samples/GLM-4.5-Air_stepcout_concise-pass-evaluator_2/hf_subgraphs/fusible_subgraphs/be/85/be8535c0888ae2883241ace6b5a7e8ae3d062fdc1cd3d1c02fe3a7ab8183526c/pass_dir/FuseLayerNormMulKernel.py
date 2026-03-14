import torch
import triton
import triton.language as tl

# Pattern matching function for the multiplication sequence
def pattern(x, weight, norm_factor):
    # Match: add operation (already done before calling this pattern)
    #        then multiplication with normalization factor
    #        then multiplication with weight
    tmp_9 = x * norm_factor  # Apply normalization scaling
    tmp_10 = weight * tmp_9  # Apply layer weight
    return tmp_9, tmp_10  # Return both intermediate and final result

# Argument extraction function
def replacement_args(x, weight, norm_factor):
    return (x, weight, norm_factor)

# Define the optimized multiplication kernel
@triton.jit
def fused_mul_kernel(
    x_ptr,  # Input tensor [batch, seq, features]
    weight_ptr,  # Weight tensor [features]
    norm_ptr,  # Normalization factors [batch, seq]
    out_ptr,  # Final output after weight multiplication
    intermediate_ptr,  # Intermediate result after normalization
    n_batch,  # Batch size
    n_seq,    # Sequence length  
    n_features,  # Feature dimension (1024)
    BLOCK_SIZE: tl.constexpr,
):
    # Program id maps to each feature in output
    pid = tl.program_id(0)
    
    if pid >= n_features:
        return
    
    # Compute thread indices for batch and sequence
    batch_seq_start = tl.arange(0, n_batch * n_seq)
    
    # Load weight for this feature dimension
    weight_val = tl.load(weight_ptr + pid)
    
    # Load normalization factors for all batch*seq combinations  
    norm_factors = tl.load(norm_ptr + batch_seq_start)
    
    # Load input values for this feature across all batch*seq
    x_vals = tl.load(x_ptr + batch_seq_start * n_features + pid)
    
    # Compute intermediate result: x * norm_factor
    intermediate_vals = x_vals * norm_factors
    
    # Compute final result: intermediate * weight
    output_vals = intermediate_vals * weight_val
    
    # Store results
    tl.store(intermediate_ptr + batch_seq_start * n_features + pid, intermediate_vals)
    tl.store(out_ptr + batch_seq_start * n_features + pid, output_vals)

# Kernel wrapper
@torch.fx.wrap
def fused_multiplications(x, weight, norm_factor):
    n_batch, n_seq, n_features = x.shape
    
    # Create output tensors
    intermediate = torch.empty_like(x, dtype=torch.float32, device=x.device)
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    # Launch kernel for feature-wise computation
    fused_mul_kernel[(n_features,)](
        x_ptr=x,
        weight_ptr=weight,
        norm_ptr=norm_factor,
        out_ptr=output,
        intermediate_ptr=intermediate,
        n_batch=n_batch,
        n_seq=n_seq,
        n_features=n_features,
        BLOCK_SIZE=1024,
    )
    
    return intermediate, output

# Replacement function
def replacement_func():
    return fused_multiplications