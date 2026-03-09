import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern matching for scalar exponential operation - key optimization target"""
    tmp_0 = in_0
    tmp_1 = tmp_0.exp()
    return tmp_1

def replacement_args(in_0):
    """Extract arguments for the replacement function"""
    return (in_0,)

# High-performance Triton kernel for scalar exponential operation
@triton.jit
def optimized_scalar_exp_kernel(
    x_ptr,
    out_ptr,
    numel: tl.constexpr,
):
    """Optimized Triton kernel for exponential computation"""
    # Each program handles one element for scalar operations
    pid = tl.program_id(0)
    if pid >= numel:
        return
    
    # Load scalar value
    x = tl.load(x_ptr + pid)
    
    # Compute exponential using Triton's built-in operations
    # Note: For scalar values, exp is computed using math operations
    out = tl.exp(x)
    
    # Store result
    tl.store(out_ptr + pid, out)

@torch.fx.wrap
def optimized_exp_performance(x):
    """
    High-performance exponential function optimized for NVIDIA A30 GPU
    Uses Triton kernel for better utilization of GPU resources
    """
    # Handle scalar input (from weight_meta.py: shape=[])
    if x.numel() == 1:
        # Use optimized Triton kernel for scalar exponential
        result = torch.empty_like(x)
        numel = x.numel()
        
        optimized_scalar_exp_kernel[(numel,)](
            x_ptr=x,
            out_ptr=result,
            numel=numel,
        )
        return result
    else:
        # Fallback for non-scalar inputs
        return x.exp()

def replacement_func():
    """Return the optimized exponential function"""
    return optimized_exp_performance