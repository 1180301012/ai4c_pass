import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern: specific to graph 1 with split [1, 196] and view [1, 384, 14, 14]"""
    # Element-wise addition
    tmp_0 = in_1 + in_0
    
    # Split with specific parameters for graph 1
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    
    # Get both parts
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    
    # Permute last two dimensions
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    # View with specific parameters for graph 1  
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    
    # Return both observable outputs
    return tmp_2, tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple addition kernel
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap 
def optimized_pattern1_wrapper(in_0, in_1):
    # Simple optimized addition with correct output shapes
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    hidden_dim = in_0.shape[2]
    
    # Optimized addition
    n_elements = in_0.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    tmp_0 = torch.empty_like(in_0)
    optimized_add_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=tmp_0,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    
    # Correctly reshape the outputs
    tmp_2 = tmp_0[:, :1, :]  # First part of split
    tmp_5 = tmp_0[:, 1:, :].transpose(1, 2).reshape(1, 384, 14, 14)
    
    return tmp_2, tmp_5

def replacement_func():
    return optimized_pattern1_wrapper