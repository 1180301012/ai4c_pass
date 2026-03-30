import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Add None dimension at position 2
    tmp_7 = input_tensor[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    # Expand to [1, 1, 8, 3, 256]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    # Reshape to [1, 8, 3, 256]
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    return tmp_9

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def expand_reshape_kernel(
    input_ptr, output_ptr,
    total_input_elements, replication_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block in the input tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_input_elements
    
    # Load input data (this will be broadcast to all heads)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # For each output head, copy the input data
    # We have replication_factor (8) heads to copy to
    for head_idx in range(replication_factor):
        # Calculate output offset for this head
        # Each head is sequential in memory: [head0_seq, head1_seq, ..., head7_seq]
        output_offset = offsets + head_idx * total_input_elements
        output_mask = output_offset < (total_input_elements * replication_factor)
        
        # Store the same data in all heads
        tl.store(output_ptr + output_offset, input_data, mask=output_mask)

@torch.fx.wrap  
def expand_reshape_optimized(input_tensor):
    # Input shape: [1, 1, 3, 256] = 768 elements total
    total_input_elements = input_tensor.numel()
    replication_factor = 8  # We need to replicate to 8 heads
    expected_output_elements = total_input_elements * replication_factor
    
    # Create output tensor with correct shape [1, 8, 3, 256]
    output = torch.empty(1, 8, 3, 256, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimal block size for this workload
    BLOCK_SIZE = 256
    
    # Calculate grid dimensions - we need one program per block of input data
    num_programs = (total_input_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    expand_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        total_input_elements=total_input_elements,
        replication_factor=replication_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return expand_reshape_optimized