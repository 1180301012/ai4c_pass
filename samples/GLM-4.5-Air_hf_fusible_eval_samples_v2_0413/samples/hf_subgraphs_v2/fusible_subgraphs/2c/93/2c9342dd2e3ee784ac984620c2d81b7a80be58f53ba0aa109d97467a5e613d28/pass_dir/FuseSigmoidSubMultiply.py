import torch
import triton
import triton.language as tl

# Pattern matching function for sigmoid → subtract 0.25 → multiply by pi fusion
def pattern(x):
    """Match the sequence: sigmoid(x) → subtract 0.25 → multiply by pi"""
    tmp_5 = x.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

# Argument extraction function
def replacement_args(x):
    # Extract input tensor
    return (x,)

# Fusion kernel using Triton with dtype support
@triton.jit
def sigmoid_subtract_multiply_kernel_fp32(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that fuses sigmoid, subtract, and multiply operations (fp32 only)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: sigmoid(x) - 0.25) * pi
    sigmoid_x = tl.sigmoid(x)
    sigmoid_minus_025 = sigmoid_x - 0.25
    result = sigmoid_minus_025 * 3.141592653589793
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Extended kernel that supports fp16/bf16 by converting to fp32
@triton.jit
def sigmoid_subtract_multiply_kernel_extended(
    input_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Extended kernel supporting fp16/bf16 through fp32 conversion"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if dtype == tl.bfloat16:
        # Convert bfloat16 to fp32 for sigmoid computation
        x_fp32 = x.to(tl.float32)
        sigmoid_x = tl.sigmoid(x_fp32)
        sigmoid_minus_025 = sigmoid_x - 0.25
        result = sigmoid_minus_025 * 3.141592653589793
        # Convert back to bfloat16
        result = result.to(tl.bfloat16)
    elif dtype == tl.float16:
        # Convert float16 to fp32 for sigmoid computation  
        x_fp32 = x.to(tl.float32)
        sigmoid_x = tl.sigmoid(x_fp32)
        sigmoid_minus_025 = sigmoid_x - 0.25
        result = sigmoid_minus_025 * 3.141592653589793
        # Convert back to float16
        result = result.to(tl.float16)
    else:
        # For fp32, direct computation
        sigmoid_x = tl.sigmoid(x)
        sigmoid_minus_025 = sigmoid_x - 0.25
        result = sigmoid_minus_025 * 3.141592653589793
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper decorated with torch.fx.wrap
@torch.fx.wrap
def fused_sigmoid_subtract_multiply(x):
    """Wrapper function to launch the fused kernel with dtype-specific optimization"""
    n_elements = x.numel()
    dtype = x.dtype
    
    # Optimize block size based on tensor size and dtype
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 10000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch the appropriate kernel based on dtype
    if dtype in [torch.float32]:
        # Use optimized fp32 kernel
        sigmoid_subtract_multiply_kernel_fp32[(num_programs,)](
            input_ptr=x,
            output_ptr=output,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Use extended kernel for fp16/bf16
        dtype_const = tl.float16 if dtype == torch.float16 else (tl.bfloat16 if dtype == torch.bfloat16 else tl.float32)
        sigmoid_subtract_multiply_kernel_extended[(num_programs,)](
            input_ptr=x,
            output_ptr=output,
            n_elements=n_elements,
            dtype=dtype_const,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

# Replacement function - returns the kernel wrapper (not called)
def replacement_func():
    return fused_sigmoid_subtract_multiply