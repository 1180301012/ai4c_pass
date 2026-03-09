import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_1 = None
    tmp_3 = 1.0 + tmp_2
    tmp_2 = None
    tmp_4 = tmp_0 * tmp_3
    tmp_0 = tmp_3 = None
    return (tmp_4,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized GELU computation: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Inline constants for better performance
    sqrt_two_x = x * 0.7071067811865475  # 1/sqrt(2) ≈ 0.7071067811865475
    erf_val = tl.erf(sqrt_two_x)
    gelu_result = x * 0.5 * (1.0 + erf_val)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@triton.jit
def gelu_kernel_optimized(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized GELU computation: x * 0.5 * (1 + erf(x * 1/sqrt(2)))
    sqrt_two_x = x * 0.7071067811865475  # 1/sqrt(2) ≈ 0.7071067811865475
    erf_val = tl.erf(sqrt_two_x)
    gelu_result = x * 0.5 * (1.0 + erf_val)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    # Ensure input is on CUDA
    if x.device.type != 'cuda':
        x = x.cuda()
    
    N = x.numel()
    
    # Choose optimal block size based on input size
    if N < 10000:
        BLOCK_SIZE = 256
    elif N < 100000:
        BLOCK_SIZE = 512
    elif N < 1000000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, device=x.device)
    
    # Use the optimized kernel with the appropriate block size
    gelu_kernel_optimized[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return triton_gelu