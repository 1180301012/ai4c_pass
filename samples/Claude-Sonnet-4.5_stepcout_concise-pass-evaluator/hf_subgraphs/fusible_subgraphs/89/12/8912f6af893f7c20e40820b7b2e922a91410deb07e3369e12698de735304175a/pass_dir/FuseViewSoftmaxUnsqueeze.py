import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Match the pattern: view -> softmax -> unsqueeze
    This pattern appears after conv2d in all target graphs.
    """
    # Get symbolic dimensions using size() to make pattern match across different shapes
    batch_size = input_tensor.size(0)
    channels = input_tensor.size(1)
    
    # View to flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
    viewed = input_tensor.view(batch_size, channels, -1)
    
    # Softmax on the last dimension (dim=2)
    softmaxed = torch.nn.functional.softmax(viewed, 2, _stacklevel=5)
    
    # Unsqueeze to add a dimension at the end: (N, C, H*W) -> (N, C, H*W, 1)
    output = softmaxed.unsqueeze(-1)
    
    return output


def replacement_args(input_tensor):
    """Extract arguments needed for the replacement function"""
    return (input_tensor,)


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized softmax kernel that processes one row per program.
    Each row represents one softmax computation.
    """
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointers for this row
    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride
    
    # Generate column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the entire row
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(x, axis=0)
    
    # Compute exp(x - max(x))
    numerator = tl.exp(x - row_max)
    
    # Compute sum of exponentials
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    softmax_output = numerator / denominator
    
    # Store the result
    tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)


@torch.fx.wrap
def fused_view_softmax_unsqueeze(input_tensor):
    """
    Fused implementation of view -> softmax -> unsqueeze.
    This reduces memory traffic by computing softmax directly on the reshaped tensor.
    """
    # Get input shape and compute dimensions
    original_shape = input_tensor.shape
    N = original_shape[0]
    C = original_shape[1]
    
    # Calculate the flattened spatial dimension
    if len(original_shape) == 4:
        H, W = original_shape[2], original_shape[3]
        HW = H * W
    elif len(original_shape) == 3:
        # Already in (N, C, HW) format
        HW = original_shape[2]
    else:
        raise ValueError(f"Unexpected input shape: {original_shape}")
    
    # Reshape input to (N, C, HW) for softmax computation
    reshaped = input_tensor.contiguous().view(N, C, HW)
    
    # Allocate output tensor
    output = torch.empty_like(reshaped)
    
    # Configure kernel launch parameters
    n_rows = N * C  # Total number of softmax operations
    n_cols = HW     # Size of each softmax
    
    # Use next power of 2 for BLOCK_SIZE for better performance
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Cap at maximum supported block size
    BLOCK_SIZE = min(BLOCK_SIZE, 131072)
    
    # Launch kernel with one program per row
    grid = (n_rows,)
    
    fused_softmax_kernel[grid](
        reshaped,
        output,
        n_rows,
        n_cols,
        reshaped.stride(0) if reshaped.dim() == 3 else reshaped.stride(1),
        output.stride(0) if output.dim() == 3 else output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Unsqueeze to add dimension at the end: (N, C, HW) -> (N, C, HW, 1)
    output = output.unsqueeze(-1)
    
    return output


def replacement_func():
    """Return the replacement function (not called, just returned)"""
    return fused_view_softmax_unsqueeze