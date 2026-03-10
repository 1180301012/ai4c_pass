import torch
import triton
import triton.language as tl

def pattern(x):
    # Reshape and transpose operations
    reshaped = x.reshape(16, -1, 64)
    transposed = reshaped.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@triton.jit
def fused_reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    input_batch_size,
    input_seq_len,
    input_hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one output element
    head_idx = pid // (input_seq_len * 64)
    seq_idx = (pid // 64) % input_seq_len
    col_idx = pid % 64
    
    # Create mask for bounds checking
    head_mask = head_idx < 16
    seq_mask = seq_idx < input_seq_len
    col_mask = col_idx < 64
    
    # Combined mask
    mask = (head_mask & seq_mask) & col_mask
    
    if mask:
        # Calculate input offsets
        input_offset = (head_idx * input_seq_len * 64 + 
                       seq_idx * 64 + 
                       col_idx)
        
        # Load input element
        input_val = tl.load(input_ptr + input_offset)
        
        # Store output in transposed position  
        output_offset = (head_idx * input_seq_len * 64 + 
                        seq_idx * 64 + 
                        col_idx)
        tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def fused_reshape_transpose_forward(x):
    input_shape = x.shape
    
    # Calculate dimensions after reshape(16, -1, 64)
    if len(input_shape) == 3:
        # Input is (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = input_shape
        # reshape(16, -1, 64) implies hidden_dim = 16*64 = 1024
        new_seq_len = hidden_dim // 64
    else:
        # Input is 2D
        batch_size, total_elements = input_shape
        # Reshape to (16, -1, 64)
        new_seq_len = total_elements // (16 * 64)
        batch_size = 1  # Assume batch size 1 for consistency
    
    # Output after transpose(1, 2): (batch, 64, new_seq_len)
    output_shape = (batch_size, 64, new_seq_len)
    total_elements = batch_size * 64 * new_seq_len
    
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    fused_reshape_transpose_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        input_batch_size=batch_size,
        input_seq_len=new_seq_len,
        input_hidden_dim=16 * 64,  # 1024
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.squeeze(0) if batch_size == 1 else output

def replacement_func():
    return fused_reshape_transpose_forward