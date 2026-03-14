import torch
import triton
import triton.language as tl

# Precise GeLU computation pattern matching
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Ultra-optimized kernel with precise computation
@triton.jit
def precise_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Optimized memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Precise constants for GeLU calculation
    cubic_const = 0.044715
    tanh_const = 0.7978845608028654
    linear_const = 0.5
    
    # Precise computation pipeline
    x_cubed = x * x * x
    cubic_term = x_cubed * cubic_const
    inner_linear = x + cubic_term
    tanh_input = inner_linear * tanh_const
    
    # Precise tanh computation using exponential identity
    exp_2x = tl.exp(2.0 * tanh_input)
    tanh_value = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Final precise computation
    gelu_inner = 1.0 + tanh_value
    output = x * linear_const * gelu_inner
    
    # Optimized store operation
    tl.store(out_ptr + offsets, output, mask=mask)

# Ultra-optimized wrapper with intelligent block sizing
@torch.fx.wrap
def precise_gelu(x):
    n_elements = x.numel()
    dtype = x.dtype
    device = x.device
    
    # Intelligent block sizing for maximum performance
    tensor_size = n_elements
    
    if tensor_size < 1 * 1024 * 1024:
        BLOCK_SIZE = 256
    elif tensor_size < 20 * 1024 * 1024:
        BLOCK_SIZE = 512
    elif tensor_size < 100 * 1024 * 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Calculate optimal grid
    grid_size = (tensor_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel with optimized configuration
    precise_gelu_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=tensor_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return precise_gelu