import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match: padding operation"""
    tmp_3 = torch.nn.functional.pad(x, (0, 0, 0, 1), 'constant', None)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_pad_kernel(
    input_ptr,
    output_ptr,
    input_rows,
    input_cols,
    pad_rows,
    pad_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized padding kernel - specifically for adding row padding"""
    row_stride = input_cols  # Each row has 'input_cols' elements
    
    # Calculate row and column indices
    row_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    col_offset = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle row index and column index within row
    row_idx = row_id
    col_idx = col_offset
    
    # Check bounds for input data
    row_mask = (row_idx < input_rows) & (col_idx < input_cols)
    
    # Calculate output position (accounting for padding)
    output_row = row_idx + (1 if row_idx >= input_rows else 0)  # Add padding for rows beyond input
    output_col = col_idx
    
    output_offset = output_row * input_cols + output_col
    
    # Load input data (masked)
    input_offset = row_idx * row_stride + col_idx
    input_val = tl.load(input_ptr + input_offset, mask=row_mask, other=0.0)
    
    # Check if we're in the padding region
    padding_row = (row_idx >= input_rows) & (col_idx < input_cols)
    padding_val = tl.where(padding_row, 0.0, input_val)
    
    # Store output (if within output bounds)
    output_size = (input_rows + pad_rows) * input_cols
    output_mask = output_offset < output_size
    tl.store(output_ptr + output_offset, padding_val, mask=output_mask & ~padding_row)

@torch.fx.wrap
def optimized_pad(x):
    """Optimized padding operation"""
    input_rows, input_cols = x.shape[-2:]  # Get last two dimensions
    pad_rows, pad_cols = 1, 0  # We're adding 1 row, 0 columns
    
    output_shape = list(x.shape)
    output_shape[-2] = input_rows + pad_rows  # Add padding to row dimension
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # For simplicity, use a 2D grid approach
    row_blocks = (input_rows + pad_rows)
    col_blocks = (input_cols + 7) // 8  # Use 8-element blocks for coalescing
    
    # Use simple 1D kernel for better performance
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create a simple kernel that handles padding efficiently
    @triton.jit
    def simple_pad_kernel(
        input_ptr,
        output_ptr,
        n_input_elements,
        input_total_elements,
        output_total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        input_mask = offsets < n_input_elements
        output_mask = offsets < output_total_elements
        
        # Load input data (zero-pad if out of bounds)
        x = tl.load(input_ptr + offsets, mask=input_mask, other=0.0)
        
        # Store output - this handles the padding automatically
        # because output buffer is larger than input buffer
        tl.store(output_ptr + offsets, x, mask=output_mask)
    
    simple_pad_kernel[(num_programs,)](
        x,
        output,
        n_input_elements,
        x.numel(),
        output.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_pad