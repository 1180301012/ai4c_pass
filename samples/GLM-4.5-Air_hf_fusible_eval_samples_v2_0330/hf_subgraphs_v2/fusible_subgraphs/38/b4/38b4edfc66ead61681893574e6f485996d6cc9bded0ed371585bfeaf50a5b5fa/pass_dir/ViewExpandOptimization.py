import torch
from torch import device
import triton
import triton.language as tl

@triton.jit
def expand_optimization_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one element in the output
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    if linear_idx >= batch_size * seq_len * seq_len:
        return
    
    # Calculate indices for [batch_size, 1, seq_len, seq_len]
    batch_idx = linear_idx // (seq_len * seq_len)
    row_idx = (linear_idx % (seq_len * seq_len)) // seq_len
    col_idx = linear_idx % seq_len
    
    # Load input value (assume input is [batch_size, seq_len])
    input_val = tl.load(input_ptr + batch_idx * seq_len + row_idx)
    
    # Store expanded value
    tl.store(output_ptr + linear_idx, input_val, mask=True)

@torch.fx.wrap
def optimized_expand_operation(input_tensor):
    """Optimized expand operation that avoids intermediate tensors"""
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    
    # Create expanded output directly
    output = torch.empty((batch_size, 1, seq_len, seq_len), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # For small sequences, use efficient broadcasting
    # This avoids multiple intermediate tensor operations
    if seq_len <= 16:
        # Use efficient expand for small sequences
        output = input_tensor.unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)
    else:
        # Use Triton kernel for larger sequences
        output_flat = output.view(batch_size * seq_len * seq_len)
        grid = (triton.cdiv(batch_size * seq_len * seq_len, 256),)
        expand_optimization_kernel[grid](
            input_ptr=input_tensor.data_ptr(),
            output_ptr=output_flat.data_ptr(),
            batch_size=batch_size,
            seq_len=seq_len,
            BLOCK_SIZE=256
        )
    
    return output

def pattern(tmp_7, tmp_8, tmp_10, tmp_11):
    """Pattern matching view and expand operations for optimization"""
    # This matches the common pattern: view -> expand operations
    # which are used to create the attention mask structure
    
    result = tmp_8.expand(1, 1, tmp_7.shape[0], tmp_7.shape[0])
    return result

def replacement_args(tmp_7, tmp_8, tmp_10, tmp_11):
    """Extract the tensor for optimized expansion"""
    return (tmp_8,)

def replacement_func():
    """Return the optimized expansion function"""
    return optimized_expand_operation