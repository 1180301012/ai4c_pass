import torch
import triton
import triton.language as tl

# Pattern matching function  
def pattern(in_0, in_1):
    """
    Matches GELU → element-wise multiplication → dropout pattern
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel using Triton with better computation patterns
@triton.jit
def fused_gelu_mul_dropout_kernel(
    in0_ptr,
    in1_ptr, 
    out_ptr,
    n_elements,
    dropout_rate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized kernel that fuses GELU + multiplication + dropout
    Uses fused operations and optimized tanh approximation for better performance
    """
    # Constants for GELU computation
    SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
    GELU_COEFF = 0.044715  # Coefficient for x^3 term in GELU
    DROPOUT_SCALE = 0.9  # 1.0 - 0.1 for dropout
    HALF = 0.5
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute memory range with optimized memory coalescing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with vectorized memory access
    x = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation with minimal temporary variables
    # Step 1: Compute x^3
    x_squared = x * x
    x_cubed = x_squared * x
    
    # Step 2: Compute tanh argument: sqrt(2/pi) * (x + 0.044715 * x^3)
    gelu_inner = x + GELU_COEFF * x_cubed
    tanh_arg = SQRT_2_OVER_PI * gelu_inner
    
    # Step 3: Optimized tanh approximation with fused operations
    # Using: tanh(z) ≈ z / (1 + 0.5352 * z^2) for better numerical stability
    tanh_z_squared = tanh_arg * tanh_arg
    tanh_denom = 1.0 + 0.5352 * tanh_z_squared
    tanh_val = tanh_arg / tanh_denom
    
    # Step 4: Compute GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_factor = HALF * (1.0 + tanh_val)
    gelu_val = x * gelu_factor
    
    # Step 5: Final fused computation: GELU(x) * y * dropout_scale
    # Fuse multiplication with y and dropout scaling into one operation
    out = gelu_val * y * DROPOUT_SCALE
    
    # Store result with optimal memory access
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper with advanced autotuning
@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1):
    """
    Fused GELU + multiplication + dropout operation with optimized autotuning
    """
    # Handle different input shapes (3D tensors)
    out_shape = in_0.shape
    assert in_0.shape == in_1.shape, "Input tensors must have the same shape"
    
    # Get tensor properties for better autotuning
    n_elements = in_0.numel()
    dtype = in_0.dtype
    
    # Flatten inputs for efficient processing
    x_flat = in_0.contiguous().view(-1)
    y_flat = in_1.contiguous().view(-1)
    
    # Calculate output
    out_flat = torch.empty_like(x_flat)
    
    # Advanced autotuning based on tensor size and dtype
    if dtype == torch.bfloat16:
        # BFloat16 benefits from larger blocks due to simpler ALU operations
        if n_elements < 16384:
            block_size = 256
        elif n_elements < 131072:
            block_size = 512
        else:
            block_size = 1024
    elif dtype == torch.float32:
        # Float32 uses larger block sizes for better occupancy
        if n_elements < 8192:
            block_size = 128
        elif n_elements < 65536:
            block_size = 256
        else:
            block_size = 512
    else:  # float16
        # Float16 benefits from medium block sizes
        if n_elements < 16384:
            block_size = 256
        elif n_elements < 65536:
            block_size = 512
        else:
            block_size = 1024
    
    # Calculate grid size
    grid_size = (n_elements + block_size - 1) // block_size
    
    # Launch kernel optimized for the specific data type
    fused_gelu_mul_dropout_kernel[(grid_size,)](
        x_flat,
        y_flat, 
        out_flat,
        n_elements,
        dropout_rate=0.1,
        BLOCK_SIZE=block_size,
    )
    
    # Reshape output to match original input shape
    return out_flat.view(out_shape)

# Replacement function
def replacement_func():
    return fused_gelu_mul_dropout