import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    """Match transpose pattern exactly as in model.py"""
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr, 
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Simplified transpose kernel for [batch, seq_len, hidden] -> [batch, hidden, seq_len]"""
    # Each program handles one element in the output [batch, hidden, seq_len]
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    # Calculate input offset: [batch, seq_len, hidden]
    input_offset = (pid_batch * seq_len + pid_seq) * hidden_size + pid_hidden
    
    # Calculate output offset: [batch, hidden, seq_len]  
    output_offset = (pid_batch * hidden_size + pid_hidden) * seq_len + pid_seq
    
    # Load from input and store to output with transposed layout
    # Since we access valid memory based on program IDs, we don't need mask + other
    val = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    """Optimized transpose implementation"""
    original_shape = input_tensor.shape
    
    # Only handle the 3D transpose case: [batch, seq_len, hidden] -> [batch, hidden, seq_len]
    if len(original_shape) != 3:
        # Fall back to regular transpose for non-3D tensors
        return input_tensor.transpose(-1, -2)
    
    batch_size, seq_len, hidden_size = original_shape
    
    # Calculate grid dimensions - one program per element in output tensor
    grid = (batch_size, hidden_size, seq_len)
    
    # Create output tensor with transposed shape
    output_shape = (batch_size, hidden_size, seq_len)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    transpose_kernel[grid](
        input_tensor,
        output,
        batch_size,
        seq_len,
        hidden_size,
        1  # BLOCK_SIZE = 1 since each program handles one element
    )
    
    return output

def replacement_func():
    return optimized_transpose