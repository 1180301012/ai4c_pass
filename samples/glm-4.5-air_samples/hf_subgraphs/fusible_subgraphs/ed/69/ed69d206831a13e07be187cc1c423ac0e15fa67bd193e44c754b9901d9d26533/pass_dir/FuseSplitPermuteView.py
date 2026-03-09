import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple addition pattern - we know this works
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fusion_kernel(
    # Input
    input_ptr,
    # Output 1: first slice ([1, 1, C])
    output1_ptr,
    # Output 2: second slice after permute and view ([1, C, H, W])
    output2_ptr,
    # Input dimensions
    N,         # Second dimension (197 or 577)
    C,         # Third dimension (384)
    split_idx,  # Index to split at (1)
    H,         # Height dimension (14 or 24)
    W,         # Width dimension (14 or 24)
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Handle each output in a separate program
    if pid == 0:
        # Output 1: [1, 1, C] - extract first slice
        # Launch multiple programs for this output to handle C dimension
        c_offset = tl.program_id(1) * BLOCK_SIZE
        mask = c_offset + tl.arange(0, BLOCK_SIZE) < C
        
        # Calculate address in flattened tensor
        input_offset = split_idx * N * C + c_offset
        output_offset = c_offset
        
        # Load from input slice and store to output
        vals = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE), mask=mask)
        tl.store(output1_ptr + output_offset + tl.arange(0, BLOCK_SIZE), vals, mask=mask)
    
    elif pid == 1:
        # Output 2: [1, C, H, W] - split, permute and view
        # Create a flattened view of [C, H, W]
        total_elements = C * H * W
        c_offset = tl.program_id(1) * BLOCK_SIZE
        mask = c_offset + tl.arange(0, BLOCK_SIZE) < total_elements
        
        # Convert offset to [c, h, w] coordinates
        flat_idx = c_offset + tl.arange(0, BLOCK_SIZE)
        w = flat_idx % W
        h = (flat_idx // W) % H
        c = flat_idx // (W * H)
        
        # Map to input: [split_idx + h, w, c] -> [split_idx + h*W + w, c]
        input_offset = (split_idx + h * W + w) * N + c
        
        # Load transposed data
        vals = tl.load(input_ptr + input_offset, mask=mask)
        tl.store(output2_ptr + flat_idx, vals, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # Perform the addition first
    tmp_0 = y + x
    
    # Now handle the split + permute + view operations
    N, C = tmp_0.shape[1], tmp_0.shape[2]
    split_idx = 1  # Fixed split at first element
    
    # Calculate output shapes
    output1_shape = (1, 1, C)
    N_split = N - split_idx
    H = int(N_split ** 0.5)  # Assuming square (14x14 or 24x24)
    W = H
    output2_shape = (1, C, H, W)
    
    # Allocate outputs
    output1 = torch.zeros(output1_shape, dtype=tmp_0.dtype, device=tmp_0.device)
    output2 = torch.zeros(output2_shape, dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Launch kernel for the full computation
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions for both outputs
    # For output 1: Need programs to handle C dimension
    num_programs_output1 = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    # For output 2: Need programs to handle C*H*W dimension  
    num_programs_output2 = (C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for each output
    # Output 1
    grid1 = (1, num_programs_output1)
    fusion_kernel[grid1](
        tmp_0,
        output1,
        output2,
        N, C, split_idx, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Output 2
    grid2 = (1, num_programs_output2)
    fusion_kernel[grid2](
        tmp_0,
        output1,
        output2,
        N, C, split_idx, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the same structure as the original computation
    return output1, output2

def replacement_func():
    return triton_add