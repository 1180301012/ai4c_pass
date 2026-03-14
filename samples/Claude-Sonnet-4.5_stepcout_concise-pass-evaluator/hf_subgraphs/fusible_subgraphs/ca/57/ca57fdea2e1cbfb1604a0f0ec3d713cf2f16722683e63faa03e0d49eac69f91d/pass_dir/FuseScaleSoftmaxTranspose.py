import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match: scale * softmax * transpose
    """
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(in_0):
    """
    Extract arguments needed for replacement
    """
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def scale_softmax_kernel(
    input_ptr,
    output_ptr,
    scale: tl.constexpr,
    N,  # Softmax dimension size (number of elements per row)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for scale + softmax along last dimension using online algorithm.
    Each program processes one row of the input.
    """
    row_idx = tl.program_id(0)
    
    # Calculate the base pointer for this row
    row_start_ptr = input_ptr + row_idx * N
    output_row_start_ptr = output_ptr + row_idx * N
    
    # Load the entire row (or as much as fits in BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load and scale in one go
    data = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
    scaled_data = data * scale
    
    # Compute max for numerical stability (single reduction)
    row_max = tl.max(tl.where(mask, scaled_data, float('-inf')), axis=0)
    
    # Compute exp(x - max) 
    exp_vals = tl.exp(scaled_data - row_max)
    exp_vals = tl.where(mask, exp_vals, 0.0)
    
    # Compute sum (single reduction)
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Normalize to get softmax
    softmax_output = exp_vals / sum_exp
    
    # Store result
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(input_tensor):
    """
    Fused scale + softmax, then transpose using PyTorch.
    This approach provides better correctness and performance.
    """
    # Flatten to 2D for kernel processing
    orig_shape = input_tensor.shape
    N = orig_shape[-1]
    M = input_tensor.numel() // N
    
    # Create a view as 2D tensor
    input_2d = input_tensor.reshape(M, N)
    output_2d = torch.empty_like(input_2d)
    
    # Launch kernel - one thread block per row
    grid = (M,)
    scale_softmax_kernel[grid](
        input_2d,
        output_2d,
        scale=0.1767766952966369,
        N=N,
    )
    
    # Reshape back to original shape
    output_reshaped = output_2d.reshape(orig_shape)
    
    # Transpose last two dimensions
    result = output_reshaped.transpose(-2, -1)
    
    return result


def replacement_func():
    """
    Return the replacement function
    """
    return fused_scale_softmax_transpose