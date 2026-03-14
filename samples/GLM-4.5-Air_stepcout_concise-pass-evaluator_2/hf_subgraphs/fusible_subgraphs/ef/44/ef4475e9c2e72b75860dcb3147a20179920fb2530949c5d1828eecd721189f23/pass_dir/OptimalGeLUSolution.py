import torch
import triton
import triton.language as tl

# Optimal GeLU computation pattern matching
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

# Final optimized kernel with maximum performance and precision
@triton.jit
def optimal_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Optimized memory access with maximum throughput
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Precomputed constants for optimum performance
    cubic_const = 0.044715
    tanh_const = 0.7978845608028654  # sqrt(2/pi)
    linear_const = 0.5
    
    # Optimized computation pipeline
    x_cubed = x * x * x  # More efficient than torch.pow
    cubic_term = x_cubed * cubic_const
    inner_linear = x + cubic_term
    tanh_input = inner_linear * tanh_const
    
    # Optimized tanh using exponential identity: tanh(x) = (e^(2x) - 1)/(e^(2x) + 1)
    exp_2x = tl.exp(2.0 * tanh_input)
    tanh_value = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Final fused computation
    gelu_inner = 1.0 + tanh_value
    output = x * linear_const * gelu_inner
    
    # Optimized store with coalesced memory access
    tl.store(out_ptr + offsets, output, mask=mask)

# Optimal kernel wrapper with adaptive block sizing
@torch.fx.wrap
def optimal_gelu(x):
    n_elements = x.numel()
    dtype = x.dtype
    device = x.device
    
    # Intelligent block sizing for maximum GPU utilization
    tensor_size = n_elements
    
    if tensor_size < 512 * 1024:
        BLOCK_SIZE = 256  # Small tensors: maximum parallelism
    elif tensor_size < 16 * 1024 * 1024:
        BLOCK_SIZE = 512  # Medium tensors: balanced approach
    elif tensor_size < 128 * 1024 * 1024:
        BLOCK_SIZE = 1024  # Large tensors: optimum throughput
    else:
        BLOCK_SIZE = 2048  # Very large tensors: maximum efficiency
    
    # Calculate optimal grid configuration
    grid_size = (tensor_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with optimal type
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel with optimal configuration
    optimal_gelu_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=tensor_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimal_gelu