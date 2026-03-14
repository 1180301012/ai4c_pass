import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Create the pattern without relying on torch.arange that causes issues with symbolic tracing
    # Instead, we'll match the mathematical structure: result = input + (arange(end) * scale)
    
    # Create the index vector on CPU first, then move it to CUDA
    # This avoids the symbolic tracing issue with torch.arange
    import torch.nn.functional as F
    
    # Get the shape of in_0
    shape = in_0.shape
    total_elements = shape[0] * shape[1]
    
    # Create index vector that repeats for each position in input
    if total_elements > 0:
        # Generate arange vector on CPU first
        cpu_range = torch.arange(0, in_2, device=torch.device('cpu'))
        # Broadcast it to match input shape pattern
        # The pattern is: for each batch position, we have arange(0, batch_size) * scale
        batch_size = in_2
        # Create pattern that repeats every batch_size elements
        index_pattern = torch.arange(total_elements, device=torch.device('cpu')) % batch_size
        scaled_pattern = index_pattern * in_1
        
        # Ensure it's on CUDA device
        if in_0.device.type != 'cuda':
            scaled_pattern = scaled_pattern.cuda()
        
        # Add to input
        result = in_0 + scaled_pattern
        return result.view(-1)
    else:
        return in_0.view(-1)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_index_vector_kernel(
    input_ptr,                    # L_index_indices tensor
    scale_ptr,                    # L_index_num_segments (scalar)
    output_ptr,                   # Output tensor
    input_x, input_y,            # Input dimensions
    batch_size,                  # in_2 value
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (input_x * input_y)
    
    # Load input tensor (L_index_indices)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Compute index_vector = arange(0, batch_size) * scale
    # We need to compute this efficiently for each position
    index_vector_tl = tl.arange(0, batch_size)
    scale_val = tl.load(scale_ptr)
    
    # Generate the pattern for all positions in input tensor
    if input_x * input_y > 0:
        # Replicate the index_vector to match input tensor size
        # The pattern repeats every input_y elements
        pos_in_batch = offsets % batch_size
        scaled_vector = pos_in_batch * scale_val
    else:
        scaled_vector = 0
    
    # Add input tensor + scaled index vector
    result = input_vals + scaled_vector
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_index_vector_addition(in_0, in_1, in_2):
    # Get input dimensions and compute total elements
    input_x, input_y = in_0.shape
    total_elements = input_x * input_y
    
    # Set Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input
    output = torch.empty_like(in_0)
    
    # Ensure inputs are on the same device
    if in_0.device.type != 'cuda':
        in_0 = in_0.cuda()
        in_2 = in_2.cuda()
    
    # Launch Triton kernel
    fused_index_vector_kernel[(num_programs,)](
        input_ptr=in_0,
        scale_ptr=in_1,
        output_ptr=output,
        input_x=input_x,
        input_y=input_y,
        batch_size=in_2.item(),  # Convert scalar value
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_index_vector_addition