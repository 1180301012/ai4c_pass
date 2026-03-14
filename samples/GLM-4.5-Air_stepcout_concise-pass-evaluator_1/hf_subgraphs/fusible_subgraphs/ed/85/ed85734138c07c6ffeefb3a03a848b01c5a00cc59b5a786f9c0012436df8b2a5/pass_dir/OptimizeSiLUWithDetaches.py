import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern that matches SiLU + detach operations"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the optimized kernel"""
    return in_0, in_1, in_2

@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Optimized SiLU kernel using Triton
    
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    """
    # Program identifier
    pid = tl.program_id(0)
    
    # Offset for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Fast sigmoid approximation for better performance
    sigmoid_x = tl.where(
        x > 0,
        1.0 / (1.0 + tl.exp(-x)),
        tl.exp(x) / (1.0 + tl.exp(x))
    )
    
    # Compute SiLU: x * sigmoid(x)
    out = x * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_silu_with_detaches(in_0, in_1, in_2):
    """Optimized implementation with custom SiLU kernel"""
    # Apply optimized SiLU to in_0
    N = in_0.numel()
    BLOCK_SIZE = 1024  # Good balance between occupancy and memory efficiency
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    tmp_0 = torch.empty_like(in_0)
    
    silu_kernel[num_programs](in_0, tmp_0, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Keep detach operations as they are (very cheap)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_func():
    """Return the optimized implementation"""
    return optimized_silu_with_detaches