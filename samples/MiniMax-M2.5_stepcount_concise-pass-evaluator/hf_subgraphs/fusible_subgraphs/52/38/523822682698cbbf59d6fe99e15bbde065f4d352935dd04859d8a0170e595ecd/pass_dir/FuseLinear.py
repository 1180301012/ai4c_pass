import torch
import triton
import triton.language as tl


def pattern(in_6, tmp_5, tmp_4):
    """
    Pattern to match: torch.nn.functional.linear(in_6, weight, bias)
    - in_6: input tensor [batch, in_features]
    - tmp_5: weight tensor [out_features, in_features]
    - tmp_4: bias tensor [out_features]
    """
    result = torch.nn.functional.linear(in_6, tmp_5, tmp_4)
    return result


def replacement_args(in_6, tmp_5, tmp_4):
    return (in_6, tmp_5, tmp_4)


# Simple Triton kernel using manual reduction to avoid dot size constraints
@triton.jit
def triton_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_o, stride_om,
):
    """
    Triton kernel for linear layer: Y = X @ W^T + b
    
    Each program handles one element of the output matrix.
    Uses manual reduction to avoid tl.dot size constraints.
    """
    # Block size for reduction
    BLOCK_SIZE_K: tl.constexpr = 256
    
    # Get which output element this program handles
    pid = tl.program_id(0)
    
    # Calculate row and column
    row = pid // N
    col = pid % N
    
    # Bounds check
    if row >= M or col >= N:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Manual reduction over K
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load input element: input[row, k + offset]
        input_ptrs = input_ptr + row * stride_im + k_offsets * stride_ik
        x = tl.load(input_ptrs, mask=k_mask, other=0.0)
        
        # Load weight elements: weight[col, k + offset]
        # weight is [N, K], so we access: weight[col, k + offset]
        weight_ptrs = weight_ptr + col * stride_wn + k_offsets * stride_wk
        w = tl.load(weight_ptrs, mask=k_mask, other=0.0)
        
        # Element-wise multiply and sum
        acc += tl.sum(x * w)
    
    # Add bias
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + col)
    
    # Store output
    tl.store(output_ptr + row * stride_o + col * stride_om, acc)


@torch.fx.wrap
def triton_linear(input, weight, bias):
    """
    Optimized linear layer using Triton with manual reduction.
    """
    M = input.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    # Total number of output elements = M * N
    grid = (M * N,)
    
    triton_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output


def replacement_func():
    return triton_linear