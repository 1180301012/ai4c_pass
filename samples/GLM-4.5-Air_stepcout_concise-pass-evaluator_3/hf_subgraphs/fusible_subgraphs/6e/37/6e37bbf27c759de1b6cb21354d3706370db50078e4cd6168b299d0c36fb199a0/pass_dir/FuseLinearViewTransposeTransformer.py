import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    # Original computation exactly as provided in model.py
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
    tmp_0 = None
    tmp_2 = tmp_1.view((in_2.shape[0], in_2.shape[1], -1, 128))
    tmp_1 = None
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_2 = None
    tmp_4 = in_1.unsqueeze(1)
    tmp_5 = in_3.unsqueeze(1)
    return (tmp_4, tmp_5, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_view_transpose_kernel(
    hidden_states_ptr,
    weight_ptr, 
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    output_inner_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Global program ID
    pid = tl.program_id(0)
    
    # Number of elements in output tensor
    total_elements = batch_size * 4 * seq_len * 128
    
    # Compute the start index for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)
    
    # Process each element in the block
    for idx in range(start_idx, end_idx):
        # Convert linear index to 4D coordinates [b, i, p, j]
        b = idx // (4 * seq_len * 128)
        remaining = idx % (4 * seq_len * 128)
        i = remaining // (seq_len * 128)  # i in [0, 3] (the 4 dimension)
        remaining = remaining % (seq_len * 128)
        p = remaining // 128  # sequence position
        j = remaining % 128   # position within 128 vector
        
        # Initialize accumulator
        acc = 0.0
        
        # Perform matrix multiplication: sum over hidden dimension
        for k in range(512):  # output dimension is 512
            # Calculate input hidden state offset [b, p, k]
            hidden_offset = b * seq_len * hidden_dim + p * hidden_dim + k
            
            # Calculate weight offset [k, i*128 + j]
            weight_offset = k * 512 + i * 128 + j
            
            # Load values
            if hidden_offset < batch_size * seq_len * hidden_dim:
                hidden_val = tl.load(hidden_states_ptr + hidden_offset)
            else:
                hidden_val = 0.0
                
            if weight_offset < 512 * 512:
                weight_val = tl.load(weight_ptr + weight_offset)
            else:
                weight_val = 0.0
                
            acc += hidden_val * weight_val
        
        # Store result at output [b, i, p, j]
        output_offset = idx
        tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def fused_linear_view_transpose(in_0, in_1, in_2, in_3):
    # Get tensor shapes
    batch_size = in_2.shape[0]  # in_2 is hidden_states
    seq_len = in_2.shape[1]     # in_2 is hidden_states  
    hidden_dim = in_2.shape[2]  # in_2 is hidden_states
    
    # Output shape after view and transpose: [batch_size, 4, seq_len, 128]
    output_shape = (batch_size, 4, seq_len, 128)
    output = torch.empty(output_shape, dtype=torch.bfloat16, device=in_0.device)  # Use in_0 device
    output_flat = output.view(-1)  # Flatten for processing
    
    # Calculate total number of elements
    total_elements = batch_size * 4 * seq_len * 128
    
    # Set up Triton kernel launch configuration
    BLOCK_SIZE = 1024  # Number of elements per program
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_linear_view_transpose_kernel[(num_programs,)](
        hidden_states_ptr=in_2,    # in_2 is hidden_states
        weight_ptr=in_0,           # in_0 is weight
        output_ptr=output_flat,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_inner_dim=128,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle unsqueeze operations for observable outputs
    tmp_4 = in_1.unsqueeze(1)   # in_1 is cos
    tmp_5 = in_3.unsqueeze(1)   # in_3 is sin
    
    return (tmp_4, tmp_5, output)

def replacement_func():
    def wrapper(in_0, in_1, in_2, in_3):
        return fused_linear_view_transpose(in_0, in_1, in_2, in_3)
    return wrapper