import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(y_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one output element
    pid = tl.program_id(0)
    # Compute memory offset
    offsets = pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute GELU activation using erf: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    y = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)



@triton.heuristics({
    'BLOCK_SIZE': lambda args: 128 if args['N'] <= 2048 else 256,
})
@triton.jit
def efficient_gelu_flatten_kernel(out_ptr, x_ptr, batch_size, channels, N, BLOCK_SIZE: tl.constexpr):
    """
    Highly optimized kernel for GELU + flatten on shapes [batch, channels, 1, 1]
    Uses vectorized operations and optimized memory access for small to medium workloads.
    """
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input data directly (memory layout is already contiguous)
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Use exact GELU using erf for maximum accuracy and better performance than sigmoid approximation
    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    y_values = x_values * 0.5 * (1.0 + tl.erf(x_values * 0.7071067811865476))
    
    # Store results
    tl.store(out_ptr + offsets, y_values, mask=mask)

@torch.fx.wrap
def fused_gelu_flatten(x):
    """
    Optimized fusion of GELU + flatten for input shapes [batch, channels, 1, 1]
    Uses autotuned kernel for optimal performance.
    """
    # Input shape: [batch_size, channels, 1, 1]
    batch_size, channels, height, width = x.shape
    assert height == 1 and width == 1, f"Expected height and width to be 1, got {height}, {width}"
    
    # Output shape: [batch_size, channels]
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    N = batch_size * channels
    
    # Let autotuning handle the optimal block size
    # For our workloads (1280 and 163840 elements), this will automatically choose:
    # - BLOCK_SIZE = 128 for small workloads (N <= 2048) 
    # - BLOCK_SIZE = 256 for larger workloads
    num_programs = (N + 128 - 1) // 128 if N <= 2048 else (N + 256 - 1) // 256
    
    efficient_gelu_flatten_kernel[(num_programs,)](
        out_ptr=out,
        x_ptr=x,
        batch_size=batch_size,
        channels=channels,
        N=N,
    )
    
    return out

def pattern(x):
    # Match the exact computation from model.py
    tmp_0 = torch.nn.functional.gelu(x, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_gelu_flatten