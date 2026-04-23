import torch
import triton
import triton.language as tl


@triton.jit
def triton_linear_bias_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused linear kernel: output = input @ weight.T + bias
    - input: [B, K]
    - weight: [N, K] (stored row-major)
    - bias: [N]
    - output: [B, N]
    """
    # Grid: one program per output element
    # Calculate which output element this program computes
    pid = tl.program_id(0)
    
    # Calculate batch and output dimension indices
    b_idx = pid // N
    n_idx = pid % N
    
    # Bounds checking
    if b_idx >= B or n_idx >= N:
        return
    
    # Compute the dot product for this (batch, output) pair
    # We need: sum_k input[b, k] * weight[n, k]
    acc = 0.0
    
    # Process in blocks to handle large K
    for k_start in range(0, K, BLOCK_SIZE):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE)
        mask = k_offsets < K
        
        # Load input row: input[b, k]
        input_val = tl.load(input_ptr + b_idx * K + k_offsets, mask=mask, other=0.0)
        
        # Load weight row: weight[n, k]
        weight_val = tl.load(weight_ptr + n_idx * K + k_offsets, mask=mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(input_val * weight_val)
    
    # Add bias
    bias_val = tl.load(bias_ptr + n_idx)
    result = acc + bias_val
    
    # Store result
    tl.store(output_ptr + b_idx * N + n_idx, result)


@torch.fx.wrap
def triton_linear_bias(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear operation: output = input @ weight.T + bias
    """
    B, K = input.shape
    N = weight.shape[0]
    
    # Allocate output
    output = torch.empty((B, N), device=input.device, dtype=input.dtype)
    
    # Grid: B * N programs (one per output element)
    grid = (B * N,)
    
    # Block size for K dimension
    BLOCK_SIZE = 256
    
    triton_linear_bias_kernel[grid](
        input, weight, bias, output,
        B, N, K,
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_6: torch.Tensor, in_5: torch.Tensor, in_4: torch.Tensor):
    """
    Match the linear operation: linear = torch.nn.functional.linear(in_6, in_5, in_4)
    where in_5 is weight [1000, 384] and in_4 is bias [1000]
    """
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear


def replacement_args(in_6: torch.Tensor, in_5: torch.Tensor, in_4: torch.Tensor):
    return (in_6, in_5, in_4)


def replacement_func():
    return triton_linear_bias