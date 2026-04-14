import torch
import triton
import triton.language as tl
import numpy as np
from torch import device

def pattern(in_0, in_1, in_2, in_3):
    """Optimize the index selection and tensor manipulation sequence"""
    tmp_4 = torch.arange(0, 9, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_6 = tmp_5
    tmp_7 = tmp_6.view(-1)
    tmp_8 = in_1.index_select(0, tmp_7)
    tmp_9 = tmp_8.view(1, 9, in_1.shape[1])
    tmp_10 = tmp_9.detach()
    tmp_11 = tmp_10.to(device(type='cuda', index=0))
    # Return the final output from this subgraph + one intermediate that needs to be preserved
    return tmp_9, tmp_11

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_index_select_kernel(
    in_1_ptr, 
    pos_offset,  # constant offset (2 in this case)
    selected_indices_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that combines index select, view, and unnecessary detaches/transfers"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate the actual indices (arange(0,9) + 2)
    indices = tl.load(selected_indices_ptr + offsets, mask=mask)
    indices = indices + pos_offset
    
    # Load the selected positions directly
    selected_data = tl.load(in_1_ptr + indices * in_1_ptr.shape[1:] + tl.arange(0, in_1_ptr.shape[1:]), mask=mask)
    
    # Store the result
    tl.store(out_ptr + offsets, selected_data, mask=mask)

@torch.fx.wrap
def optimized_index_selection_and_manipulation(in_1, hidden_size):
    """Optimized function that combines index selection and tensor manipulation"""
    # Create indices and add offset directly
    indices = torch.arange(0, 9, dtype=torch.int64, device=in_1.device) + 2
    
    # Calculate output size
    output_size = 9 * hidden_size
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty((9, hidden_size), dtype=in_1.dtype, device=in_1.device)
    
    # Update kernel to handle 2D indexing properly
    @triton.jit
    def optimized_kernel_2d(
        in_1_ptr,
        indices_ptr,
        out_ptr,
        hidden_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (9 * hidden_size)
        
        # Calculate which element and which hidden dimension
        elem_idx = offsets // hidden_size
        hidden_idx = offsets % hidden_size
        
        # Load index then position
        actual_index = tl.load(indices_ptr + elem_idx, mask=elem_idx < 9)
        pos_offset = actual_index * hidden_size + hidden_idx
        
        # Load the selected data
        selected_data = tl.load(in_1_ptr + pos_offset, mask=mask)
        tl.store(out_ptr + offsets, selected_data, mask=mask)
    
    optimized_kernel_2d[(num_programs,)](
        in_1_ptr=in_1,
        indices_ptr=indices,
        out_ptr=out,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final format [1, 9, hidden_size]
    return out.view(1, 9, hidden_size)

def replacement_func():
    return optimized_index_selection_and_manipulation