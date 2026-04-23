import torch
import triton
import triton.language as tl


# =============================================================================
# Triton kernel for Linear operation: y = x @ W^T + b
# Input: [M, K], Weight: [N, K], Bias: [N]
# Output: [M, N]
# =============================================================================


@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_in: tl.constexpr,
):
    """
    Triton kernel for linear layer.
    Each program handles one row of the output [M, N].
    """
    pid = tl.program_id(0)
    row_idx = pid
    
    # Compute offsets for this row
    row_offset = row_idx * stride_in
    
    # Offsets for N dimension (512)
    n_offsets = tl.arange(0, 512)
    n_mask = n_offsets < N
    
    # Accumulator for output row [N]
    acc = tl.zeros((512,), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(K):
        # Load one element from input row at position k
        input_val = tl.load(input_ptr + row_offset + k, mask=k < K)
        
        # Load weight column k: [N] elements
        weight_col_ptr = weight_ptr + k * N
        w_vals = tl.load(weight_col_ptr + n_offsets, mask=n_mask)
        
        # Multiply and accumulate
        acc += input_val * w_vals
    
    # Add bias
    bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask)
    acc = acc + bias_vals
    
    # Store output [N]
    output_offset = row_idx * N
    tl.store(output_ptr + output_offset + n_offsets, acc, mask=n_mask)


def pattern(in_5, in_1, in_0):
    """
    Pattern: linear(in_5, in_1, in_0)
    Returns: linear result
    """
    result = torch.nn.functional.linear(in_5, in_1, in_0)
    return result


def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)


@torch.fx.wrap
def linear_wrapper(in_5, in_1, in_0):
    """
    Triton-based linear wrapper.
    """
    M = in_5.shape[0]  # 300
    K = in_5.shape[1]  # 256
    N = in_1.shape[0]  # 512
    
    # Allocate output [M, N]
    output = torch.empty((M, N), dtype=in_5.dtype, device=in_5.device)
    
    # Grid: one program per row
    grid = (M,)
    
    # Ensure contiguous memory
    input_cont = in_5.contiguous()
    weight_cont = in_1.contiguous()
    bias_cont = in_0.contiguous()
    
    linear_kernel[grid](
        input_cont, weight_cont, bias_cont, output,
        M, N, K, in_5.stride(0),
    )
    
    return output


def replacement_func():
    return linear_wrapper