import torch
import triton
import triton.language as tl
import math

# Pattern matching function for layer normalization
def pattern(tmp_2, in_1, in_0):
    """
    Pattern: Match the layer normalization operation
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    """
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_4

# Argument extraction function
def replacement_args(tmp_2, in_1, in_0):
    return (tmp_2, in_1, in_0)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr,           # Input tensor (hidden states) 
    gamma_ptr,       # Weight tensor
    beta_ptr,        # Bias tensor
    out_ptr,         # Output tensor
    n_feats,         # Feature dimension size (1024)
    n_seqs,          # Sequence dimension size
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for sequence positions
    pid_seq = tl.program_id(0)
    
    # Starting offset for this sequence position
    seq_offset = pid_seq * n_feats
    
    # Create offsets within this sequence
    offsets = seq_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_feats
    
    # Load the slice for this sequence position
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean for this sequence position
    active_elements = tl.sum(mask)
    mean = tl.sum(x) / active_elements
    
    # Compute variance using numerically stable approach
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / active_elements
    
    # Compute inverse standard deviation with epsilon protection
    # Use epsilon protection to avoid division by zero or very small numbers
    std_inv = tl.rsqrt(variance + eps)
    
    # Load gamma and beta (use float32 for better numerical stability)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Convert inputs to float32 for computation stability
    x_float = x.to(tl.float32)
    mean_float = mean.to(tl.float32)
    
    # Perform normalization with float32 precision
    x_centered = x_float - mean_float
    x_norm = x_centered * std_inv
    y = gamma * x_norm + beta
    
    # Convert back to original dtype and store
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)

@torch.fx.wrap
def triton_layer_norm(x, weight, bias, normalized_shape=1024, eps=1e-05):
    """
    Wrapper function for optimized layer normalization kernel
    """
    # Get tensor dimensions
    n_feats = x.shape[-1]  # 1024 (feature dimension)
    n_seqs = x.numel() // n_feats  # Sequence count
    
    # Use optimized block size for features
    if n_feats == 1024:
        BLOCK_SIZE = 1024  # Process full feature dimension
    else:
        BLOCK_SIZE = 256   # Default smaller block size
    
    # Calculate grid dimensions (one program per sequence)
    num_programs = n_seqs
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch Triton kernel
    optimized_layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        gamma_ptr=weight,
        beta_ptr=bias,
        out_ptr=out,
        n_feats=n_feats,
        n_seqs=n_seqs,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_layer_norm