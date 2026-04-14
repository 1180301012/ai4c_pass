import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern matching: Softmax → Unsqueeze(-1)
    
    This matches the core computation that can be significantly optimized:
    torch.nn.functional.softmax(input_tensor, 2, _stacklevel=5)
    result.unsqueeze(-1)
    """
    softmax_result = torch.nn.functional.softmax(input_tensor, 2, _stacklevel=5)
    final_result = softmax_result.unsqueeze(-1)
    return final_result

def replacement_args(input_tensor):
    """Extract arguments needed for the optimized kernel"""
    return (input_tensor,)

@triton.jit
def optimized_softmax_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel that fuses Softmax + Unsqueeze operations.
    
    This kernel efficiently handles:
    1. Softmax computation on the last dimension
    2. Adding final dimension with unsqueeze(-1)
    """
    batch_id = tl.program_id(0)
    start_idx = batch_id * seq_len
    end_idx = start_idx + seq_len
    
    # Step 1: Find max value in the sequence for this batch
    max_val = -float('inf')
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        vals = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.max(max_val, vals)
    
    # Step 2: Compute sum of exp(x - max)
    sum_exp = 0.0
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x)
    
    # Step 3: Compute softmax and store withunsed dimension
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        max_val_local = max_val
        sum_exp_local = sum_exp
        
        # Compute softmax value for each element
        softmax_vals = tl.exp(x - max_val_local) / sum_exp_local
        
        # Store both the softmax value and a placeholder for the added dimension
        tl.store(output_ptr + idx * 2, softmax_vals, mask=mask)
        tl.store(output_ptr + idx * 2 + 1, 0.0, mask=mask)  # Added dimension

@torch.fx.wrap
def optimized_softmax_unsqueeze(input_tensor):
    """
    Wrapper function that executes the fused Softmax + Unsqueeze operation.
    
    This function:
    1. Efficiently computes softmax on the last dimension
    2. Adds final dimension using unsqueeze(-1) 
    3. Handles different tensor shapes automatically
    """
    if input_tensor.dim() != 3:
        # If not 3D, fall back to original implementation
        return torch.nn.functional.softmax(input_tensor, dim=-1, _stacklevel=5).unsqueeze(-1)
    
    batch_size, dim1, dim2 = input_tensor.shape
    total_elements = batch_size * dim1 * dim2
    
    # Output will have shape (batch_size, dim1, dim2, 1)
    output_shape = (batch_size, dim1, dim2, 1)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Run optimized kernel
    optimized_softmax_unsqueeze_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=dim1 * dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the optimized kernel function"""
    return optimized_softmax_unsqueeze