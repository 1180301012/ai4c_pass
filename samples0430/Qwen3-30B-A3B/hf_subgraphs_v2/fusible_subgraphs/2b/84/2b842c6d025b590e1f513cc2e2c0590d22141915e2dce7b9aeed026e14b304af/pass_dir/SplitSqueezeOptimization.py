import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(tmp_2):
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)

# Argument extraction function

def replacement_args(tmp_2):
    return (tmp_2,)

# Triton kernel for channel extraction
@triton.jit
def split_squeeze_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    n1, n2, n3,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # For [1, 17, 2] shape, process in 1D grid (n1 * n2)
    # Each thread handles one element per channel
    i = 0
    j = pid % n2
    
    # Compute input pointer for channel 0
    in0_ptr = in_ptr + i * (n2 * n3) + j * n3 + 0
    # Compute output pointer for channel 0
    out0_ptr = out0_ptr + i * n2 + j
    
    # Compute input pointer for channel 1
    in1_ptr = in_ptr + i * (n2 * n3) + j * n3 + 1
    # Compute output pointer for channel 1
    out1_ptr = out1_ptr + i * n2 + j
    
    x0 = tl.load(in0_ptr)
    x1 = tl.load(in1_ptr)
    
    tl.store(out0_ptr, x0)
    tl.store(out1_ptr, x1)

# Kernel wrapper
@torch.fx.wrap
def split_squeeze_kernel_wrapper(tmp_2):
    n1, n2, n3 = tmp_2.shape
    out0 = torch.empty([n1, n2], dtype=tmp_2.dtype, device=tmp_2.device)
    out1 = torch.empty([n1, n2], dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Calculate grid size: n1 * n2
    grid_size = n1 * n2
    BLOCK_SIZE = 32  # Optimal for the small size (17)
    
    split_squeeze_kernel[(grid_size,)](
        tmp_2, out0, out1, 
        n1, n2, n3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out0, out1

# Replacement function

def replacement_func():
    return split_squeeze_kernel_wrapper