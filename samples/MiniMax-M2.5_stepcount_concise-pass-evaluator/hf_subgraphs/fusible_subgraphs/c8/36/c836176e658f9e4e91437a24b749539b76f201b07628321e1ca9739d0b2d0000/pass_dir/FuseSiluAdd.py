import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the pattern: silu(in_1) + in_0
    This fuses silu activation and element-wise addition into one kernel.
    """
    tmp = torch.nn.functional.silu(in_1)
    out = tmp + in_0
    return out

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel that fuses silu and add
@triton.jit
def silu_add_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    # Using stable computation: x * sigmoid(x) = x / (1 + exp(-x))
    # But for better numerical stability, we use: x * sigmoid(x)
    # where sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -in_1
    sigmoid = 1.0 / (1.0 + tl.exp(neg_x))
    silu_out = in_1 * sigmoid
    
    # Compute silu(in_1) + in_0
    out = silu_out + in_0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def silu_add_kernel_autotuned(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU activation: x * sigmoid(x)
    neg_x = -in_1
    sigmoid = 1.0 / (1.0 + tl.exp(neg_x))
    silu_out = in_1 * sigmoid
    
    # Compute silu(in_1) + in_0
    out = silu_out + in_0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Autotune configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_add_kernel_optimized(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU activation: x * sigmoid(x)
    neg_x = -in_1
    sigmoid = 1.0 / (1.0 + tl.exp(neg_x))
    silu_out = in_1 * sigmoid
    
    # Compute silu(in_1) + in_0
    out = silu_out + in_0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


# Kernel wrapper with autotuning
@torch.fx.wrap
def silu_add_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that launches the optimized Triton kernel.
    Fuses silu(in_1) + in_0 into a single kernel.
    """
    # Flatten inputs for 1D parallelism
    n_elements = in_0.numel()
    
    # Allocate output
    out = torch.empty_like(in_0)
    
    # Define grid
    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    # Launch autotuned kernel
    silu_add_kernel_optimized[grid](
        in_0, in_1, out,
        n_elements,
    )
    
    return out


def replacement_func():
    return silu_add_kernel_wrapper