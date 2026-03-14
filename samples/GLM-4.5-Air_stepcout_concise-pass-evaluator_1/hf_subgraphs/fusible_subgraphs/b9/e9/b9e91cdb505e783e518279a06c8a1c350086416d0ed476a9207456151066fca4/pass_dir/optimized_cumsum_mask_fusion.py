import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # The original computation pattern from model.py
    tmp_1 = input_tensor.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_cumsum_mask_kernel(
    input_ptr, 
    output_ptr, 
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid columns
    mask = col_idx < n_cols
    
    # Load input element
    input_val = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=0)
    
    # Apply the optimized computation:
    # 1. Create boolean mask (input != 1)
    is_not_one = input_val != 1
    
    # 2. Convert to int (0 for False, 1 for True)
    int_mask = tl.where(is_not_one, 1, 0)
    
    # 3. Compute cumulative sum manually along columns
    cumsum = tl.cumsum(int_mask, 0)
    
    # 4. The +0 and type conversion happen implicitly through tl operations
    # 5. Multiply by original mask (gating operation)
    gated_cumsum = cumsum * int_mask
    
    # 6. Apply final transformations: convert to long and add 1
    final_result = tl.where(gated_cumsum > 0, gated_cumsum.to(tl.int64) + 1, 0)
    
    # Store the result
    tl.store(output_ptr + row_idx * n_cols + col_idx, final_result, mask=mask)

@torch.fx.wrap
def optimized_cumsum_mask(input_tensor):
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((n_rows, n_cols), dtype=torch.int64, device=input_tensor.device)
    
    # Set block size and grid size
    BLOCK_SIZE = 1024
    n_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    grid = (n_rows, n_blocks)
    
    optimized_cumsum_mask_kernel[grid](
        input_tensor,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_cumsum_mask