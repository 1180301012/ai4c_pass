import torch
import triton
import triton.language as tl

@triton.jit
def expand_small_kernel(
    param_ptr,    # 1D parameter tensor [D]
    out_ptr,      # 4D output [D, 1, 1]
    n_elements,   # D (number of elements in parameter)
):
    """Optimized kernel for small tensors - single work-item"""
    pid = tl.program_id(0)
    
    # Only one work-item handles all elements for small tensors
    if pid == 0:
        for i in range(n_elements):
            param_val = tl.load(param_ptr + i)
            tl.store(out_ptr + i * 1 * 1, param_val)

@triton.jit
def expand_param_kernel(
    param_ptr,    # 1D parameter tensor [D]
    out_ptr,      # 4D output [D, 1, 1]
    n_elements,   # D (number of elements in parameter)
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for parameter expansion: copying 1D values to 4D tensor"""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)
    
    # Each program handles a block of the parameter
    for i in range(start_idx, end_idx):
        # Copy parameter value to expanded position [i, 0, 0]
        param_val = tl.load(param_ptr + i)
        tl.store(out_ptr + i * 1 * 1, param_val)  # Store at [i, 0, 0]

@torch.fx.wrap
def optimized_expand_1d_to_4d_1_1(param):
    """
    Adaptive parameter expansion: [D] -> [D, 1, 1]
    Uses specialized kernels for different tensor sizes
    """
    D = param.shape[0]
    target_shape = (D, 1, 1)
    
    # Create empty tensor with correct shape
    expanded = torch.empty(target_shape, dtype=param.dtype, device=param.device)
    
    # Use different kernel strategies based on tensor size
    if D <= 64:
        # Small tensor: single work-item kernel to avoid launch overhead
        expand_small_kernel[(1,)](
            param_ptr=param,
            out_ptr=expanded,
            n_elements=D,
        )
    elif D <= 512:
        # Medium tensor: medium block size
        BLOCK_SIZE = 64
        num_programs = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
        expand_param_kernel[(num_programs,)](
            param_ptr=param,
            out_ptr=expanded,
            n_elements=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Large tensor: large block size for better GPU utilization
        BLOCK_SIZE = 256
        num_programs = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
        expand_param_kernel[(num_programs,)](
            param_ptr=param,
            out_ptr=expanded,
            n_elements=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return expanded



def pattern(param):
    """Match: Two consecutive unsqueeze(-1) operations on a 1D tensor"""
    tmp_5 = param.unsqueeze(-1)  # [D] -> [D, 1]
    tmp_6 = tmp_5.unsqueeze(-1)  # [D, 1] -> [D, 1, 1]
    return tmp_6

def replacement_args(param):
    """Extract arguments for replacement function"""
    return (param,)

def replacement_func():
    """Return the optimized expansion function"""
    return optimized_expand_1d_to_4d_1_1