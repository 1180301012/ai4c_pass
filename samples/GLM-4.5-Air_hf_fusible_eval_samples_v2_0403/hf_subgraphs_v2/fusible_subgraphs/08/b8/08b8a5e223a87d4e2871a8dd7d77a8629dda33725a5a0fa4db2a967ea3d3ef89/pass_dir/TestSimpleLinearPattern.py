import torch
import triton
import triton.language as tl

# Very simple pattern - just linear operation
def pattern(in_0, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    return (linear,)

# Argument extraction function
def replacement_args(in_0, in_3):
    return (in_0, in_3)

# Simple linear kernel - one row per kernel instance
@triton.jit
def simple_linear_kernel(
    weight_ptr, in_ptr, out_ptr,
    M, N, K,
    N_CONST: tl.constexpr,
):
    # Program id corresponds to row index
    row_idx = tl.program_id(0)
    
    if row_idx < M:
        # Initialize accumulator for this row using constant size
        row_sum = tl.zeros((N_CONST,), dtype=tl.float32)
        
        # Loop over K dimension for this row
        for k in range(0, K, 1):
            # Load the entire weight row (with bounds check)
            weight_row = tl.load(
                weight_ptr + tl.arange(0, N_CONST),
                mask=tl.arange(0, N_CONST) < N,
                other=0.0
            )
            
            # Load the specific input element
            input_val = tl.load(
                in_ptr + (row_idx * K + k),
                mask=True,
                other=0.0
            )
            
            # Accumulate: input * weight_row
            row_sum += input_val * weight_row
        
        # Store the result row with bounds check
        tl.store(
            out_ptr + (row_idx * N_CONST) + tl.arange(0, N_CONST),
            row_sum,
            mask=tl.arange(0, N_CONST) < N
        )

@torch.fx.wrap
def optimized_simple_linear(in_0, in_3):
    # Get dimensions: weight [K, N], input [M, K], output [M, N]  
    weight_shape = in_0.shape  # [K, N]
    in_shape = in_3.shape      # [M, K]
    M, K, N = in_shape[0], in_shape[2], weight_shape[1]
    
    # Create output tensor
    linear_out = torch.empty((M, N), device=in_3.device, dtype=in_3.dtype)
    
    # Launch kernel if valid dimensions
    if M > 0 and K > 0 and N > 0:
        grid = (M,)
        
        simple_linear_kernel[grid](
            in_0, in_3, linear_out, M, N, K, 512  # Use constant max size
        )
    
    return (linear_out,)

def replacement_func():
    return optimized_simple_linear