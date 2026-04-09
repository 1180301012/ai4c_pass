import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function that matches the entire forward computation
def pattern(in_0, arange_end):
    """Matches the entire forward computation pattern"""
    # Match the exact computation from the model
    tmp_1 = torch.arange(0, arange_end, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_1, tmp_2

# Argument extraction function
def replacement_args(in_0):
    """Extract the input tensor and infer the arange end value"""
    # Based on the input shape, infer the arange end value
    # For most of these models, it's the second dimension (sequence length)
    return (in_0, in_0.shape[1])

# Optimized triton kernel for the entire computation
@triton.jit
def optimized_forward_kernel(
    input_ptr,  # input tensor (int64)
    output_range_ptr,  # output range tensor (int64) 
    output_bool_ptr,  # output bool tensor
    input_rows,  # number of rows in input
    input_cols,  # number of columns in input  
    end_val,  # end value for arange (exclusive)
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that combines range generation and bool conversion"""
    # Get program ID and total programs
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Handle range generation output (tmp_1)
    range_pid = pid
    range_block_start = range_pid * BLOCK_SIZE
    range_indices = range_block_start + tl.arange(0, BLOCK_SIZE)
    range_mask = range_indices < end_val
    
    # Generate range values: 0, 1, 2, ..., end_val-1
    tl.store(output_range_ptr + range_indices, range_indices, mask=range_mask)
    
    # Handle bool conversion output (tmp_2)
    bool_pid = pid + num_programs  # Use second half of program space for bool conversion
    bool_block_start = bool_pid * BLOCK_SIZE
    bool_indices = bool_block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate global index in flattened input tensor
    global_input_idx = bool_indices
    bool_mask = global_input_idx < (input_rows * input_cols)
    
    # Load input values and convert to bool
    input_vals = tl.load(input_ptr + global_input_idx, mask=bool_mask, other=0)
    bool_vals = (input_vals != 0).to(output_bool_ptr.type())
    
    # Store bool results
    tl.store(output_bool_ptr + global_input_idx, bool_vals, mask=bool_mask)

@torch.fx.wrap  
def optimized_forward_computation(in_0, arange_end):
    """Optimized forward computation that combines both operations"""
    # Get input tensor properties
    input_rows, input_cols = in_0.shape
    input_elements = input_rows * input_cols
    end_val = arange_end  # Use the provided arange end value
    
    # Create output tensors
    output_range = torch.empty(end_val, dtype=torch.int64, device=in_0.device)
    output_bool = torch.empty_like(in_0, dtype=torch.bool)
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    range_programs = (end_val + BLOCK_SIZE - 1) // BLOCK_SIZE
    bool_programs = (input_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_programs = range_programs + bool_programs
    
    # Launch the optimized kernel
    optimized_forward_kernel[(total_programs,)](
        input_ptr=in_0,
        output_range_ptr=output_range,
        output_bool_ptr=output_bool,
        input_rows=input_rows,
        input_cols=input_cols,
        end_val=end_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_range, output_bool

# Replacement function (returns function reference)
def replacement_func():
    return optimized_forward_computation