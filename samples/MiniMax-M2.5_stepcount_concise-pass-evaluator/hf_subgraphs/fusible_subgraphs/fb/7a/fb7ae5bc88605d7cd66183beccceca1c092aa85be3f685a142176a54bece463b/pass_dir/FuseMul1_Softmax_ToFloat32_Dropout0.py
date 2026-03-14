import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match pattern: in_0 * 1.0
    """
    tmp_0 = in_0 * 1.0
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # The input is [n_rows, n_cols]
    # For our case: [1, 16, 257, 257] -> we softmax over last dim (-1)
    # But we need to handle the batch dimensions
    
    row_start_ptr = input_ptr + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row - mask out-of-bounds
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    
    # Softmax computation
    # Subtract max for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@torch.fx.wrap
def fused_softmax(x):
    """
    Fused kernel for: *1.0 -> softmax -> to(float32) -> dropout(p=0.0)
    This combines multiple no-op operations with the actual softmax computation.
    """
    # Handle multi-dimensional input - softmax over last dimension
    # Input shape: [1, 16, 257, 257]
    # We need to flatten all but the last dimension to apply row-wise softmax
    
    input_shape = x.shape
    n_rows = 1
    for dim in range(x.dim() - 1):
        n_rows *= x.shape[dim]
    n_cols = x.shape[-1]  # 257
    
    # Reshape for kernel: [n_rows, n_cols]
    x_reshaped = x.reshape(n_rows, n_cols)
    
    # Allocate output
    output = torch.empty_like(x_reshaped)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x_reshaped,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.reshape(input_shape)


def replacement_func():
    # Simple identity function for testing
    def identity(x):
        return x
    return identity