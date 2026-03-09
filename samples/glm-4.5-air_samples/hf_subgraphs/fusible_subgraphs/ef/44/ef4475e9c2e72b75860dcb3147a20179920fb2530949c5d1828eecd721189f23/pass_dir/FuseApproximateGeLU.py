import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = 0.5 * x
    tmp_1 = torch.pow(x, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_1 = None
    tmp_3 = x + tmp_2
    tmp_2 = None
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_3 = None
    tmp_5 = torch.tanh(tmp_4)
    tmp_4 = None
    tmp_6 = 1.0 + tmp_5
    tmp_5 = None
    tmp_7 = tmp_0 * tmp_6
    tmp_0 = tmp_6 = None
    return tmp_7

def replacement_args(x):
    return (x,)

@triton.jit
def approximate_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with vectorized memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants from the original computation
    c1 = 0.5
    c2 = 0.044715
    c3 = 0.7978845608028654
    
    # Optimized computation sequence with minimal registers
    # Compute x^3 directly and fuse all operations
    x_cubed = x * x * x
    x_plus_cubed = x + (c2 * x_cubed)
    scaled = c3 * x_plus_cubed
    
    # Optimized tanh computation using stable formula
    exp_val = tl.exp(-2.0 * scaled)
    x_tanh = (2.0 / (1.0 + exp_val)) - 1.0
    
    # Final fused computation: 0.5 * x * (1.0 + tanh_result)
    out = (c1 * x) * (1.0 + x_tanh)
    
    # Store result with coalesced memory access
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_approximate_gelu(x):
    N = x.numel()
    
    # Optimal block size for different tensor sizes
    if N >= 1048576:  # Large tensors (> 1M elements)
        BLOCK_SIZE = 2048
    elif N >= 262144:  # Medium-large tensors
        BLOCK_SIZE = 1024
    elif N >= 65536:   # Medium tensors
        BLOCK_SIZE = 512
    else:              # Small tensors
        BLOCK_SIZE = 256
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with optimal configuration
    approximate_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_approximate_gelu