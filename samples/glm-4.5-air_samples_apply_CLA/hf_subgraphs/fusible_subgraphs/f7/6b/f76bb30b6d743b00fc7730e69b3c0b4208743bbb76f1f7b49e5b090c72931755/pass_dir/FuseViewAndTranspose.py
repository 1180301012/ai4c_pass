import torch
import triton
import triton.language as tl

def pattern(x):
    # View and transpose operations
    viewed = x.view(1, -1, 16, 64)
    transposed = viewed.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@triton.jit
def fused_view_transpose_kernel(
    input_ptr,
    output_ptr,
    input_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one head of one sequence position
    seq_idx = pid // 16
    head_idx = pid % 16
    col_idx = pid % 64  # Each program handles one column within the head
    
    # Create mask for bounds checking
    seq_mask = seq_idx < input_seq_len
    head_mask = head_idx < 16
    col_mask = col_idx < 64
    
    # Combined mask
    mask = (seq_mask & head_mask) & col_mask
    
    if mask:
        # Load input data for this sequence position and head element
        input_offset = seq_idx * 1024 + head_idx * 64 + col_idx
        input_val = tl.load(input_ptr + input_offset)
        
        # Store output with transposed dimensions
        output_offset = head_idx * input_seq_len * 64 + seq_idx * 64 + col_idx
        tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def fused_view_transpose_forward(x):
    input_shape = x.shape
    input_seq_len = input_shape[1]
    hidden_dim = input_shape[2]
    
    # Output should be (1, seq_len, 16, 64) -> transpose to (1, 16, seq_len, 64)
    output_shape = (1, 16, input_seq_len, 64)
    total_elements = input_seq_len * 16
    
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    fused_view_transpose_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        input_seq_len=input_seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the transposed result (remove batch dimension)
    return output.squeeze(0)  # Shape: (16, seq_len, 64)

def replacement_func():
    return fused_view_transpose_forward