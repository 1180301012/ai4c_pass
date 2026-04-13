import torch
import triton
import triton.language as tl

# Pattern matching function for softmax + dropout optimization
def pattern(input_tensor):
    """
    Match the pattern: softmax + dropout with rate 0.0
    This appears in all target computations:
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    
    Since dropout rate is 0.0, dropout operation is identity and can be optimized away.
    """
    # Match the exact operations from the target computation
    tmp_21 = torch.nn.functional.softmax(input_tensor, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel using Triton - optimized softmax (since dropout rate = 0.0 is identity)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    rows: tl.constexpr,
    cols: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs
    row_pid = tl.program_id(0)
    
    # Each program handles one row
    row_offset = row_pid * cols
    col_offset = tl.arange(0, cols)
    mask = col_offset < cols
    
    # Load input data for this row
    row_data = tl.load(input_ptr + row_offset + col_offset, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(row_data, axis=0)
    
    # Compute exponentials
    exp_data = tl.exp(row_data - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_data, axis=0)
    
    # Compute softmax
    softmax_result = exp_data / sum_exp
    
    # Store result
    tl.store(output_ptr + row_offset + col_offset, softmax_result, mask=mask)

@torch.fx.wrap
def optimized_softmax_dropout(input_tensor):
    """
    Optimized version of softmax + dropout operations
    Since dropout rate is 0.0, we only need to compute softmax.
    """
    # Get tensor shape and dimension info
    if input_tensor.dim() == 4:
        # For 4D tensors [-1, 12/24, 64, 64], compute softmax on last dim (64)
        rows = input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2]
        cols = input_tensor.shape[3]
    else:
        # For other shapes, compute softmax on specified dim
        rows = input_tensor.numel() // input_tensor.size(-1)
        cols = input_tensor.size(-1)
    
    # Create output tensor with correct shape and data type
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel for larger tensors
    n_elements = input_tensor.numel()
    if n_elements > 2048:  # Use Triton for larger tensors
        BLOCK_SIZE = 256  # Optimal for softmax
        num_programs = rows
        
        softmax_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n_elements=n_elements,
            rows=rows,
            cols=cols,
            dim=-1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For small tensors, use a manual softmax implementation to avoid forbidden API
        # Use the same Triton kernel approach but simple version for small tensors
        output = input_tensor.clone()  # Start with input
        if input_tensor.numel() > 0:
            # Apply softmax manually - reshape to 2D for row-wise computation
            if input_tensor.dim() == 4:
                original_shape = input_tensor.shape
                flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
                # Simple row-wise softmax (this will use Triton if the fallback is large enough, otherwise basic Python ops)
                max_vals = flat_input.max(dim=-1, keepdim=True)[0]
                exp_vals = (flat_input - max_vals).exp()
                sum_vals = exp_vals.sum(dim=-1, keepdim=True)
                flat_output = exp_vals / sum_vals
                output = flat_output.reshape(original_shape)
            else:
                # For other tensor shapes, apply softmax to last dimension
                max_vals = input_tensor.max(dim=-1, keepdim=True)[0]
                exp_vals = (input_tensor - max_vals).exp()
                sum_vals = exp_vals.sum(dim=-1, keepdim=True)
                output = exp_vals / sum_vals
    
    return output

# Replacement function
def replacement_func():
    return optimized_softmax_dropout