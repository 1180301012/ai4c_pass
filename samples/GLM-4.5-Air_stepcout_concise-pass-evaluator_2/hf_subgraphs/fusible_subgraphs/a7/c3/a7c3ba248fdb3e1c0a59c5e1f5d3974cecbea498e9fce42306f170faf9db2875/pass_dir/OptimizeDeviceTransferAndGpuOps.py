import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match the exact computation pattern from the original graph"""
    tmp_0 = in_0
    tmp_1 = tmp_0 * in_1
    tmp_0 = None
    tmp_2 = torch.as_tensor(in_2, device=device(type='cuda'))
    tmp_3 = torch.as_tensor(tmp_1, device=device(type='cuda'))
    tmp_1 = None
    tmp_4 = torch.as_tensor([-1], dtype=torch.int64)
    tmp_5 = torch.as_tensor((), dtype=torch.int64)
    tmp_6 = torch.cat([tmp_4, tmp_5], dim=0)
    tmp_4 = tmp_5 = None
    return (tmp_2, tmp_3, tmp_6)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def optimized_device_ops_forward(in_0, in_1, in_2):
    """Optimized version that eliminates unnecessary device transfers"""
    # Move scalars to GPU and perform multiplication directly on GPU
    # This avoids CPU->GPU transfer of multiplication result
    a_gpu = in_0.to('cuda')
    b_gpu = in_1.to('cuda') 
    scalar_result = a_gpu * b_gpu
    
    # Eliminate redundant device copy: in_2 is already on GPU
    # Original code does torch.as_tensor(in_2, device=device(type='cuda'))
    # which creates an unnecessary copy since in_2 is already on GPU
    tmp_2 = in_2  # Use directly, no copy needed
    
    # Eliminate tensor concatenation overhead: 
    # Original creates tmp_4 = [-1], tmp_5 = empty(), then concatenates
    # This is equivalent to just creating [-1] tensor directly
    tmp_6 = torch.tensor([-1], dtype=torch.int64, device='cuda')
    
    # All operations now optimized
    tmp_3 = scalar_result
    
    return (tmp_2, tmp_3, tmp_6)

def replacement_func():
    return optimized_device_ops_forward