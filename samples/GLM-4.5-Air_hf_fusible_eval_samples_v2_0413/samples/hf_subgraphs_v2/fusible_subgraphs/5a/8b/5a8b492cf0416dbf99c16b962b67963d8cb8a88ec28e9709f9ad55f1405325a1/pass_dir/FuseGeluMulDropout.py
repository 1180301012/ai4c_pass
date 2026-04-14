import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_mul_dropout_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    dropout_prob,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs GELU(x) * y * dropout in a single pass
    Optimized for better GPU performance
    """
    # Get program ID and compute memory offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # GELU activation: x * 0.5 * (1.0 + erf(x / sqrt(2)))
    sqrt2 = 1.41421356237
    gelu_x = x * 0.5 * (1.0 + tl.erf(x / sqrt2))
    
    # Element-wise multiplication
    mul_result = gelu_x * y
    
    # Dropout: skip dropout for now to focus on performance optimization
    # Using 1.0 multiplier means no dropout applied
    final_result = mul_result
    
    # Store result
    tl.store(out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1, dropout_prob=0.1):
    """
    Fused function that performs GELU(x) * y * dropout in one kernel
    Optimized launch configuration
    """
    # Get tensor properties
    shape = in_0.shape
    n_elements = in_0.numel()
    
    # Optimal block size for maximum GPU occupancy  
    BLOCK_SIZE = 4096  # Larger block size for better GPU utilization
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input
    dtype = in_0.dtype
    device = in_0.device
    out = torch.empty(shape, dtype=dtype, device=device)
    
    # Launch Triton kernel
    fused_gelu_mul_dropout_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        n_elements,
        dropout_prob,
        BLOCK_SIZE
    )
    
    return out

def pattern(in_0, in_1):
    """
    Pattern matching: GELU -> Multiplication -> Dropout
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement function
    """
    return (in_0, in_1)

def replacement_func():
    """
    Return the fused function as a zero-argument function
    """
    return fused_gelu_mul_dropout