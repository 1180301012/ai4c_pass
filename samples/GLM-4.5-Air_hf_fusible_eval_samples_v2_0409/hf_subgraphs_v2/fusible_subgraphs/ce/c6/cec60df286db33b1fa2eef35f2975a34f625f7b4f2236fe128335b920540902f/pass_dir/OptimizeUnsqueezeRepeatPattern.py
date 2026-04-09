import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function
def pattern():
    """Match the sequence: torch.arange -> unsqueeze(0) -> repeat(1, 1)"""
    tmp_0 = torch.arange(0, 1, device=device(type='cuda', index=0))
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return (tmp_0, tmp_2)

# Argument extraction function
def replacement_args():
    # No external arguments needed for this optimization
    return ()

@triton.jit
def create_tensors_kernel(out_1_ptr, out_2_ptr, BLOCK_SIZE: tl.constexpr):
    """Kernel that directly creates the required tensors without intermediate operations"""
    # For these tiny tensors, we just need to write single values
    pid = tl.program_id(0)
    
    if pid == 0:
        # tmp_0 should be [0] - write single 0
        tl.store(out_1_ptr + 0, 0.0)
        # tmp_2 should be [[0]] - write single 0 at position 0
        tl.store(out_2_ptr + 0, 0.0)

@torch.fx.wrap
def optimized_tensors_creation():
    """Directly create both required tensors using Triton kernel without intermediate operations"""
    # tmp_0 is a 1D tensor with [0]
    tmp_0 = torch.empty((1,), dtype=torch.float32, device='cuda:0')
    # tmp_2 is a 2D tensor with [[0]]
    tmp_2 = torch.empty((1, 1), dtype=torch.float32, device='cuda:0')
    
    BLOCK_SIZE = 1
    
    # Launch kernel with single grid program to handle both tensors
    create_tensors_kernel[(1,)](
        out_1_ptr=tmp_0,
        out_2_ptr=tmp_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_0, tmp_2

# Replacement function
def replacement_func():
    return optimized_tensors_creation