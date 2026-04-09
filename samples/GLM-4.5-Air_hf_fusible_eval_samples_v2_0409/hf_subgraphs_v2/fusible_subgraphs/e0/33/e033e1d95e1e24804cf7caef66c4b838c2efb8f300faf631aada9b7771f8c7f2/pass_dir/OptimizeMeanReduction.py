import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_4 = in_2.mean(dim = -2, keepdim = True)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple and efficient kernel for mean reduction"""
    # Each program handles one element of output (one batch, one hidden dimension)
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)
    
    # Calculate base offset for first element in this batch/hidden combination
    base_offset = batch_idx * seq_len * hidden_size + hidden_idx
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Load all sequence elements with vectorization
    offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate mask for boundary conditions
        current_offset = base_offset + i * hidden_size
        mask = (i + offset) < seq_len
        
        # Load input values - accessing [batch_idx, i, hidden_idx] for each i
        input_vals = tl.load(input_ptr + current_offset + offset * hidden_size, mask=mask, other=0.0)
        sum_val = sum_val + tl.sum(input_vals)
    
    # Calculate mean and store
    mean_val = sum_val / seq_len
    output_offset = batch_idx * hidden_size + hidden_idx
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_mean(in_2):
    # Get tensor shapes
    batch_size, seq_len, hidden_size = in_2.shape
    
    # Create output tensor (temporarily as 2D, will add singleton dim later)
    output_2d = torch.zeros((batch_size, hidden_size), dtype=in_2.dtype, device=in_2.device)
    
    # Configure kernel launch
    BLOCK_SIZE = min(256, seq_len)  # Use block size based on sequence length
    num_programs = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid_x = batch_size
    grid_y = hidden_size
    
    # Launch kernel
    optimized_mean_kernel[(grid_x, grid_y)](
        input_ptr=in_2,
        output_ptr=output_2d,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Add singleton dimension to match expected output shape
    return output_2d.view(batch_size, 1, hidden_size)



def replacement_func():
    return optimized_mean