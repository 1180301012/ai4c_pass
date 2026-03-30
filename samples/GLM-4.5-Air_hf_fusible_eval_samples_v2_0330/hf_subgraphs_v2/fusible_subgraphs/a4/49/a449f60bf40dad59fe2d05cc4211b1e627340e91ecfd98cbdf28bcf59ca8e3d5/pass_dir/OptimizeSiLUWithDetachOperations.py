import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match the exact computation pattern from model.py"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function"""
    return (in_0, in_1, in_2)

@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance SiLU kernel: x * sigmoid(x)"""
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute SiLU: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    exp_neg_x = tl.exp(-x)
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    out = x * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_silu(x):
    """High-performance SiLU implementation using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024 if N >= 1024 else 256
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, device=x.device)
    
    silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2):
    """Optimized forward pass with custom SiLU and efficient detach operations"""
    # Apply optimized SiLU
    tmp_0 = optimized_silu(in_0)
    
    # Efficient detach operations - these are already quite optimal
    # but we ensure they're done explicitly as in the original
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach() 
    tmp_3 = tmp_0.detach()
    
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_func():
    """Return the optimized function"""
    return optimized_forward