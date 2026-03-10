import torch
import triton
import triton.language as tl

def pattern(tensor_a, tensor_b, tensor_c):
    # This matches: tensor_d = torch.cat((tensor_a, tensor_b), dim=-1)
    #              result = torch.stack((tensor_d, tensor_c), dim=-1).transpose(-1, -2)
    tensor_d = torch.cat((tensor_a, tensor_b), dim=-1)
    stacked = torch.stack((tensor_d, tensor_c), dim=-1)
    result = stacked.transpose(-1, -2)
    return result

def replacement_args(tensor_a, tensor_b, tensor_c):
    return (tensor_a, tensor_b, tensor_c)

@triton.jit
def optimized_stack_transpose_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_rows,
    a_cols,
    b_cols,
    c_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Check if row is within bounds
    if row_idx >= n_rows:
        return
        
    # Calculate base addresses for this row
    a_base = row_idx * a_cols
    b_base = row_idx * b_cols
    c_base = row_idx * c_cols
    out_base = row_idx * (2 * c_cols)
    
    # Handle each element in the row separately for simplicity
    for col in range(0, c_cols, BLOCK_SIZE):
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < c_cols
        
        # Load concatenation of a and b (first half of output)
        if col < c_cols // 2:
            # Load from a
            a_offsets = a_base + col + tl.arange(0, BLOCK_SIZE)
            a_mask = a_offsets < a_cols
            a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
            
            # Store to first half of output slice 0
            out_slice0_base = out_base + col
            tl.store(out_ptr + out_slice0_base, a_vals, mask=mask)
        else:
            # Load from b  
            b_col = col - c_cols // 2
            b_offsets = b_base + b_col + tl.arange(0, BLOCK_SIZE)
            b_mask = b_offsets < b_cols
            b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
            
            # Store to second half of output slice 0
            out_slice0_base = out_base + col
            tl.store(out_ptr + out_slice0_base, b_vals, mask=mask)
        
        # Load c (second slice of output)
        c_offsets = c_base + col + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < c_cols
        c_vals = tl.load(c_ptr + c_offsets, mask=c_mask, other=0.0)
        
        # Store to output slice 1 (second half of output)
        out_slice1_base = out_base + c_cols + col
        tl.store(out_ptr + out_slice1_base, c_vals, mask=mask)

@torch.fx.wrap
def optimized_concat_stack_transpose(tensor_a, tensor_b, tensor_c):
    n_rows = tensor_a.shape[0]
    a_cols = tensor_a.shape[1]
    b_cols = tensor_b.shape[1]
    c_cols = tensor_c.shape[1]
    
    BLOCK_SIZE = 128
    num_programs = n_rows
    
    # Output shape: [n_rows, 2, c_cols]
    out_shape = [n_rows, 2, c_cols]
    out = torch.empty(out_shape, dtype=tensor_a.dtype, device=tensor_a.device)
    
    optimized_stack_transpose_kernel[(num_programs,)](
        a_ptr=tensor_a,
        b_ptr=tensor_b,
        c_ptr=tensor_c,
        out_ptr=out,
        n_rows=n_rows,
        a_cols=a_cols,
        b_cols=b_cols,
        c_cols=c_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_concat_stack_transpose