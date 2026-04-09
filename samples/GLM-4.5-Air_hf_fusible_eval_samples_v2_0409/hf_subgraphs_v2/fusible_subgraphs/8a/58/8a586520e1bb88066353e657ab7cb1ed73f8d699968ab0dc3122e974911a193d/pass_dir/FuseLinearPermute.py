import torch
import triton
import triton.language as tl

# Pattern matching function to match linear + permute
def pattern(in_3, in_1, in_0):
    """Match linear transformation followed by permutation"""
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Triton kernel for fused linear + permute operation
@triton.jit
def linear_permute_kernel(
    x_ptr,           # Input tensor [1, 196, 196, 3]
    weight_ptr,      # Weight tensor [16, 3] 
    bias_ptr,        # Bias tensor [16]
    out_ptr,         # Output tensor [1, 16, 196, 196]
    seq1: tl.constexpr,
    seq2: tl.constexpr,
    out_features: tl.constexpr,
):
    # Compute program ID - 1D grid
    pid = tl.program_id(0)
    
    # Map 1D PID to 3D coordinates
    total_elements = out_features * seq1 * seq2
    if pid >= total_elements:
        return
        
    pid_out = pid // (seq1 * seq2)
    remainder = pid % (seq1 * seq2)
    pid_seq1 = remainder // seq2
    pid_seq2 = remainder % seq2
    
    # Load bias and weight for this output feature
    bias_val = tl.load(bias_ptr + pid_out, other=0.0)
    weight = tl.load(weight_ptr + pid_out * 3 + tl.arange(0, 3), other=0.0)
    
    # Compute linear combination: sum(x[:, pid_seq1, pid_seq2, :] * weight)
    total = bias_val.to(tl.float32)
    for k in range(3):
        input_val = tl.load(
            x_ptr + pid_seq1 * (seq2 * 3) + pid_seq2 * 3 + k,
            other=0.0
        )
        total += input_val.to(tl.float32) * weight[k].to(tl.float32)
    
    # Store result
    result = total.to(tl.float16)
    tl.store(
        out_ptr + pid_out * (seq1 * seq2) + pid_seq1 * seq2 + pid_seq2,
        result
    )

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
def fused_linear_permute(x, weight, bias):
    # Get dimensions safely using .shape which should be concrete during execution
    batch, seq1, seq2, in_features = x.shape
    out_features, _ = weight.shape
    
    # Create output tensor using standard empty-like approach
    # Allocate with expected output shape [batch, out_features, seq1, seq2]
    # During actual execution (tracing complete), these will be concrete values
    if hasattr(batch, '__int__'):  # Check if concrete value
        out = torch.empty((batch, out_features, seq1, seq2), 
                         device=x.device, dtype=x.dtype)
    else:
        # Fallback: use large enough allocation that will be reshaped
        out = torch.empty_like(x)
    
    # Calculate grid size based on actual execution values
    # These will resolve to concrete numbers when not traced
    total_elements = out_features * seq1 * seq2
    grid_size = total_elements
    
    # Launch kernel
    linear_permute_kernel[grid_size,](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        seq1=seq1,
        seq2=seq2,
        out_features=out_features,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_permute