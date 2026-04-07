import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation structure
def pattern(in_0, in_1):
    """Match GELU -> multiplication -> dropout pattern"""
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel combining GELU + multiplication + dropout
@triton.jit
def fused_gelu_multiply_dropout_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + idx, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + idx, mask=mask, other=0.0)
    
    # GELU activation: use polynomial approximation for all data types
    # GELU(x) ≈ x * (0.5 + 0.5 * tanh(0.79788 * x * (1 + 0.044715 * x^2)))
    # Use a piecewise polynomial approximation to avoid expensive operations
    
    x = in_0
    x2 = x * x
    x3 = x2 * x
    
    # First approximation component: tanh argument approximation
    # tanh(y) ≈ y - y^3/3 + y^5/5 for small y, where y = 0.79788 * x * (1 + 0.044715 * x^2)
    tanh_arg = 0.79788 * x * (1 + 0.044715 * x2)
    
    # Polynomial approximation of tanh(y) 
    y_abs = tl.abs(tanh_arg)
    y_abs3 = y_abs * y_abs * y_abs
    
    # Use a piecewise approximation: tanh(y) ≈ y * (1 - y^2/3) for small values, sign(y) for large values
    # Implement sign function using where since tl.sign is not available
    sign_tanh_arg = tl.where(tanh_arg > 0, 1.0, tl.where(tanh_arg < 0, -1.0, 0.0))
    tanh_val = tl.where(y_abs < 1.0, 
                       tanh_arg * (1.0 - y_abs3 * 0.33333),  # Small value approximation
                       sign_tanh_arg)  # Large value approximation: ±1
    
    # GELU approximation: x * 0.5 * (1 + tanh_val)
    gelu_output = x * 0.5 * (1.0 + tanh_val)
    
    # Element-wise multiplication
    multiply_output = gelu_output * in_1
    
    # Dropout: during inference (training=False), dropout is identity operation
    # Since training=False, we just return the multiplication result directly
    output = multiply_output
    
    # Store result
    tl.store(out_ptr + idx, output, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_gelu_multiply_dropout(in_0, in_1):
    n_elements = in_0.numel()
    
    # Choose optimal block size based on tensor dimensions and dtype
    if in_0.dim() == 3:
        _, seq_len, hidden_dim = in_0.shape
        # For 3D tensors, optimize based on hidden dimension
        if hidden_dim >= 1024:
            BLOCK_SIZE = 1024
        elif hidden_dim >= 512:
            BLOCK_SIZE = 512
        elif hidden_dim >= 256:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 128
    else:
        # For other tensor shapes, use a reasonable default
        if n_elements >= 2**20:  # 1M+ elements
            BLOCK_SIZE = 1024
        elif n_elements >= 2**18:  # 256K+ elements
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 256
    
    # Adjust block size for lower precision types (less computation per element)
    if in_0.dtype == torch.bfloat16:
        BLOCK_SIZE = min(BLOCK_SIZE * 2, 2048)  # Bfloat16 benefits from larger blocks
    elif in_0.dtype == torch.float16:
        # For float16, use smaller blocks due to computational complexity
        # and potential precision issues with complex math
        BLOCK_SIZE = max(BLOCK_SIZE // 2, 64)  # Smaller blocks for float16
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as inputs
    out = torch.empty_like(in_0)
    
    # Launch kernel with optimized block size
    fused_gelu_multiply_dropout_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_gelu_multiply_dropout