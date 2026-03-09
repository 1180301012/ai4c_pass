import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Match: (in_0 * scale).softmax(dim=-1).transpose(-2, -1)"""
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel - no autotune
@triton.jit
def fused_softmax_transpose_kernel(
    input_ptr,
    output_ptr,
    scale: tl.constexpr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    batch_idx = row_idx // n_rows
    row = row_idx % n_rows
    
    # Calculate the starting offset for this row in input
    row_start = batch_idx * n_rows * n_cols + row * n_cols
    
    # Process columns in blocks
    exp_sum = 0.0
    
    # First pass: compute sum of exp(x*scale)
    for j in range(0, n_cols, BLOCK_SIZE_N):
        col_offsets = j + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < n_cols
        
        offsets = row_start + col_offsets
        x = tl.load(input_ptr + offsets, mask=col_mask, other=0.0)
        x = x * scale
        exp_x = tl.exp(x)
        exp_sum += tl.sum(tl.where(col_mask, exp_x, 0.0))
    
    # Second pass: compute softmax and store in transposed positions
    for j in range(0, n_cols, BLOCK_SIZE_N):
        col_offsets = j + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < n_cols
        
        offsets = row_start + col_offsets
        x = tl.load(input_ptr + offsets, mask=col_mask, other=0.0)
        x = x * scale
        exp_x = tl.exp(x)
        softmax_vals = exp_x / exp_sum
        
        # Store in transposed position: output[..., col_offsets, row]
        # Output has shape [..., n_cols, n_rows]
        out_base = batch_idx * n_rows * n_cols
        out_offsets = out_base + col_offsets * n_rows + row
        tl.store(output_ptr + out_offsets, softmax_vals, mask=col_mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0):
    """Wrapper for the fused multiply-softmax-transpose kernel"""
    scale = 0.1767766952966369
    
    shape = in_0.shape
    n_dims = len(shape)
    
    n_cols = shape[-1]  # 400
    n_rows = shape[-2]  # 400
    
    # Compute total rows to process
    if n_dims > 2:
        other_dims = 1
        for dim in range(n_dims - 2):
            other_dims *= shape[dim]
        n_rows_total = other_dims * n_rows
    else:
        n_rows_total = n_rows
    
    # Allocate output with transposed last two dims
    output_shape = list(shape)
    output_shape[-2], output_shape[-1] = output_shape[-1], output_shape[-2]
    output = torch.empty(output_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel - use block size that covers the full row (400)
    # This allows processing each row in a single pass
    BLOCK_SIZE_N = 512  # Enough to cover 400 columns
    grid = (n_rows_total,)
    fused_softmax_transpose_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        scale=scale,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper