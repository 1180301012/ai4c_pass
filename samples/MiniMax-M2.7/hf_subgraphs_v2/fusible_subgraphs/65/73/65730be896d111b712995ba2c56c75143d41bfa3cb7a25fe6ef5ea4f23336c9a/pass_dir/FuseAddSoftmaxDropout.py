import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete fused pattern
def pattern(in_0, in_1):
    """
    Pattern matches:
    1. in_1 += in_0; in_2 = in_1
    2. tmp_1 = in_2.float()
    3. tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    4. tmp_3 = tmp_2.type_as(in_2)
    5. tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    
    Returns (tmp_4,) to match the model's return structure.
    """
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_1 = None
    tmp_3 = tmp_2.type_as(in_2)
    tmp_2 = None
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    tmp_3 = None
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_softmax_kernel(
    x_ptr,         # Input pointer (after add)
    output_ptr,    # Output pointer
    n_elements,    # Total number of elements
    n_cols,        # Number of columns (softmax dimension)
    orig_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Load data
    2. Compute softmax (with numeric stability)
    3. Apply dropout (training=False is a pass-through)
    4. Store result
    """
    # Each program processes one row (all elements along softmax dim)
    row_id = tl.program_id(0)
    
    # Calculate row offsets
    row_offset = row_id * n_cols
    
    # Load all values for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_offset + col_offsets, mask=mask, other=float('-inf'))
    
    # Softmax computation with numeric stability: exp(x - max(x)) / sum(exp(x - max(x)))
    # Step 1: Find max value in the row
    max_val = tl.max(x, axis=0)
    
    # Step 2: Compute exp(x - max)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    
    # Step 3: Compute sum of exponentials
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Step 4: Compute softmax output
    softmax_out = exp_x / sum_exp
    
    # Step 5: Apply dropout (training=False means keep all values)
    # With p=0.1 and training=False, output = input (identity operation)
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that:
    1. Performs in-place add: in_1 += in_0
    2. Launches fused softmax kernel
    """
    # Perform in-place add
    in_1 += in_0
    
    # Get tensor info
    x = in_1
    orig_dtype = x.dtype
    shape = x.shape
    
    # Flatten for processing but maintain row structure for softmax
    n_rows = shape[0]
    for dim_size in shape[:-1]:
        n_rows *= dim_size
    n_cols = shape[-1]
    total_elements = n_rows * n_cols
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define block size - use 1024 for good occupancy (covers up to 1024 cols)
    BLOCK_SIZE = 1024
    
    # Launch kernel with one program per row
    num_programs = n_rows
    
    fused_softmax_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=total_elements,
        n_cols=n_cols,
        orig_dtype=orig_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper