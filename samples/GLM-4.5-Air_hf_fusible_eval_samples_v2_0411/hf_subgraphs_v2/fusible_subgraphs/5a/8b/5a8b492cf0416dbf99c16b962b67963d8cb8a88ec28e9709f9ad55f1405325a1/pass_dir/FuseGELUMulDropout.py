import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0, in_1):
    # Match the exact computation sequence from the models
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Define GELU activation function in Triton
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Compute GELU activation: GELU(x) = x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x^3)))"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU using a better polynomial approximation
    # More accurate approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(0.79788 * x * (1 + 0.044715 * x^3)))
    # But since we can't use tanh, we use a polynomial approximation that's closer to the real GELU
    # Calculate cube and use a more accurate formula
    cube = x * x * x
    cubed_term = 0.044715 * cube
    # Use a better sigmoid-like approximation for the inner function
    inner = 0.7978845608028654 * x * (1.0 + cubed_term)
    
    # Approximate tanh(x) with: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2 + x^4) for better accuracy
    x_abs = tl.where(inner > 0, inner, -inner)
    x_sq = inner * inner
    x_4th = x_sq * x_sq
    tanh_approx = inner * (27.0 + x_sq) / (27.0 + 9.0 * x_sq + x_4th)
    
    gelu_out = x * 0.5 * (1.0 + tanh_approx)
    
    # Store output
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

# Define the fused kernel that combines GELU, multiplication, and dropout
@triton.jit
def fused_gelu_mul_dropout_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. GELU activation on in_0
    2. Element-wise multiplication with in_1  
    3. Dropout with probability dropout_p
    """
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Apply GELU activation using a better polynomial approximation
    # More accurate approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(0.79788 * x * (1 + 0.044715 * x^3)))
    # But since we can't use tanh, we use a polynomial approximation that's closer to the real GELU
    # Calculate cube and use a more accurate formula
    cube = in_0 * in_0 * in_0
    cubed_term = 0.044715 * cube
    # Use a better sigmoid-like approximation for the inner function
    inner = 0.7978845608028654 * in_0 * (1.0 + cubed_term)
    
    # Approximate tanh(x) with: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2 + x^4) for better accuracy
    x_abs = tl.where(inner > 0, inner, -inner)
    x_sq = inner * inner
    x_4th = x_sq * x_sq
    tanh_approx = inner * (27.0 + x_sq) / (27.0 + 9.0 * x_sq + x_4th)
    
    gelu_out = in_0 * 0.5 * (1.0 + tanh_approx)
    
    # Step 2: Element-wise multiplication
    mul_out = gelu_out * in_1
    
    # Step 3: Apply dropout (training=False, so deterministic)
    # For inference (training=False), dropout is just a scaling operation
    # No actual masking needed, just scale to maintain expected value
    scale_factor = 1.0 / (1.0 - dropout_p) if dropout_p > 0 else 1.0
    final_out = mul_out * scale_factor
    
    # Store result
    tl.store(out_ptr + offsets, final_out, mask=mask)

# Flexible kernel wrapper that handles different data types and shapes
@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1):
    """Fused GELU + multiplication + dropout operation with automatic data type handling"""
    # Check if inputs are on CUDA (based on weight_meta.py all should be on cuda:0)
    if in_0.device.type != 'cuda' or in_1.device.type != 'cuda':
        raise ValueError("This fused implementation only supports CUDA tensors")
    
    # Determine data type from inputs (both should be same)
    dtype = in_0.dtype
    
    # Handle different data types
    if dtype == torch.float32:
        effective_dtype = torch.float32
    elif dtype in (torch.float16, torch.bfloat16):
        effective_dtype = dtype
    else:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Intelligent block size selection based on input size and data type
    if dtype == torch.float32:
        if n_elements < 1000000:
            BLOCK_SIZE = 1024
        elif n_elements < 10000000:
            BLOCK_SIZE = 2048
        else:
            BLOCK_SIZE = 4096
    else:  # float16/bfloat16 can use larger blocks
        if n_elements < 1000000:
            BLOCK_SIZE = 2048
        elif n_elements < 10000000:
            BLOCK_SIZE = 4096
        else:
            BLOCK_SIZE = 8192
    
    # Choose number of warps for better GPU utilization
    if dtype == torch.float32:
        num_warps = 4 if BLOCK_SIZE >= 1024 else 2
    else:
        num_warps = 8 if BLOCK_SIZE >= 4096 else 4
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor using allowed allocation API
    out = torch.empty_like(in_0, dtype=effective_dtype)
    
    # Launch Triton kernel on GPU with optimization parameters
    dropout_p = 0.1  # Fixed from the pattern
    fused_gelu_mul_dropout_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return out

# Replacement function - returns the fused function
def replacement_func():
    return fused_gelu_mul_dropout