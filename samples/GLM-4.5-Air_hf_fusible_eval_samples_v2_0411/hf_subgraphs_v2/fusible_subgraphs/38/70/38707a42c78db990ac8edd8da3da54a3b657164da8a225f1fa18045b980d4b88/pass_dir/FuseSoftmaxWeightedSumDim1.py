import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Match the exact computation pattern from the model:
    # softmax -> multiply by linspace -> sum along dim=1
    import torch
    from torch import device
    
    tmp_0 = torch.nn.functional.softmax(input_tensor, dim=1)
    tmp_1 = torch.linspace(0, 4, steps=5, device=device(type='cuda', index=0))
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    
    # Return the intermediate that would be returned by the model (pre-final subtraction)
    # The pattern must return the same observable intermediates as the model
    return tmp_3

def replacement_args(input_tensor):
    # We only need the input tensor for the fused computation
    return (input_tensor,)

@triton.jit
def fused_softmax_weighted_sum_kernel(
    input_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Precomputed weights [0, 1, 2, 3, 4]
    weights = tl.load(input_ptr + tl.arange(0, n_cols), mask=tl.arange(0, n_cols) < n_cols)
    
    # Row-wise parallel processing
    row_id = tl.program_id(0)
    row_offset = row_id * n_cols
    
    if row_id >= n_rows:
        return
    
    # Load input row
    input_row = tl.load(input_ptr + row_offset + tl.arange(0, n_cols), 
                      mask=tl.arange(0, n_cols) < n_cols)
    
    # Compute softmax
    max_val = tl.max(input_row)
    exp_x = tl.exp(input_row - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    
    # Compute weighted sum
    weighted_sum = tl.sum(softmax * weights, axis=0)
    
    # Store result
    tl.store(output_ptr + row_offset, weighted_sum)

@torch.fx.wrap
def fused_softmax_weighted_sum(input_tensor):
    """Fused implementation of softmax + weighted sum"""
    n_rows, n_cols = input_tensor.shape
    output = torch.empty((n_rows,), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimized block sizes for small tensor [1, 5]
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = n_cols
    
    # Grid launch
    grid = (n_rows,)
    
    fused_softmax_weighted_sum_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Returns the fused function"""
    return fused_softmax_weighted_sum