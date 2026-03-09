import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_6

def replacement_args(tmp_2):
    return (tmp_2,)



@triton.jit
def broadcast_kernel(
    input_ptr,
    output_ptr, 
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row replication - we need 3 programs total
    row_id = tl.program_id(0)
    
    # Calculate how many elements each row needs to process
    total_elements = batch_size * seq_len
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements within this row
    mask = offsets < total_elements
    
    # Load input data (same data for all rows)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Calculate output offset for this specific row
    # Row 0: 0:total_elements, Row 1: total_elements:2*total_elements, Row 2: 2*total_elements:3*total_elements
    output_base = row_id * total_elements
    output_offsets = output_base + offsets
    
    # Store the replicated data in the correct position
    tl.store(output_ptr + output_offsets, input_data, mask=mask)

@torch.fx.wrap
def direct_broadcast(tmp_2):
    batch_size = tmp_2.size(0)
    seq_len = tmp_2.size(1)
    total_elements = batch_size * seq_len
    
    # Use optimal block size based on tensor dimensions
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor [3, batch_size, seq_len]
    expanded_shape = (3, batch_size, seq_len)
    output = torch.empty(expanded_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Launch the broadcast kernel with 1D grid (3 programs, one for each row)
    # Each program handles one row replication
    broadcast_kernel[(3,)](
        input_ptr=tmp_2,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return direct_broadcast