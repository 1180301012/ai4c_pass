import torch
import triton
import triton.language as tl

# Vectorized GeLU pattern matching - fused optimized computation
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

# Ultra-optimized vectorized kernel with inline constants
@triton.jit
def vectorized_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load with stride optimization
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Inline all constants for maximum performance optimization
    # GeLU coefficients as constexpr values
    cubic_factor = 0.044715
    tanh_factor = 0.7978845608028654
    linear_factor = 0.5
    
    # Highly optimized computation pipeline
    # Compute x^3 efficiently
    x_cube = x * x * x
    # Compute cubic term
    cubic_term = x_cube * cubic_factor
    # Inner linear combination
    inner_combo = x + cubic_term
    # Tanh input phase
    tanh_phase = inner_combo * tanh_factor
    # Exponential computation using fused operation
    exp_arg = 2.0 * tanh_phase
    exp_result = tl.exp(exp_arg)
    # Optimized tanh computation
    tanh_val = (exp_result - 1.0) / (exp_result + 1.0)
    # GeLU inner component
    gelu_inner = 1.0 + tanh_val
    # Final fused result vectorization
    result = x * linear_factor * gelu_inner
    
    # Vectorized store with coalesced memory access
    tl.store(out_ptr + offsets, result, mask=mask)

# Hyper-optimized kernel wrapper with adaptive grid sizing
@torch.fx.wrap
def vectorized_gelu(x):
    n_elements = x.numel()
    dtype = x.dtype
    device = x.device
    
    # Ultra-optimized block sizing strategy
    tensor_size = n_elements
    
    if tensor_size < 512 * 1024:
        # Small tensors: maximum parallelism
        BLOCK_SIZE = 256
    elif tensor_size < 16 * 1024 * 1024:
        # Medium tensors: balanced approach
        BLOCK_SIZE = 512
    elif tensor_size < 128 * 1024 * 1024:
        # Large tensors: optimal throughput
        BLOCK_SIZE = 1024
    else:
        # Huge tensors: maximum occupancy
        BLOCK_SIZE = 2048
    
    # Calculate efficient grid configuration
    num_programs = (tensor_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with optimal alignment
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch vectorized kernel with optimal grid configuration
    vectorized_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=tensor_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return vectorized_gelu