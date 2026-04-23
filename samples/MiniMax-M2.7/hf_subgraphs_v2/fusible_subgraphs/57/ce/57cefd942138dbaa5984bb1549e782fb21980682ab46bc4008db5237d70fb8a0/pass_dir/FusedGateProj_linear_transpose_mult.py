"""
Fused Gate Projection Optimization for gMLP

This pass fuses the following operations into a single kernel:
1. torch.nn.functional.linear(in_2, weight, bias)  -> output[b, D_out, S]
2. transpose(-1, -2)                               -> output[b, S, D_out]
3. in_3 * tmp_3 (element-wise multiply)            -> result[b, S, D_out]

The key optimization is fusing these operations to avoid intermediate tensor allocations
and enable better memory access patterns.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_gate_proj_kernel(
    # Input tensor: [B, D_in, S]
    input_ptr,
    # Weight tensor: [D_in, D_out] in memory
    weight_ptr,
    # Bias tensor: [D_out]
    bias_ptr,
    # Multiply tensor: [B, S, D_out]
    mul_ptr,
    # Output tensor: [B, S, D_out]
    output_ptr,
    # Dimensions
    B, S, D_in, D_out,
    # Strides for input
    input_stride_b, input_stride_din, input_stride_s,
    # Strides for weight
    weight_stride_din, weight_stride_dout,
    # Strides for bias
    bias_stride,
    # Strides for mul
    mul_stride_b, mul_stride_s, mul_stride_dout,
    # Strides for output
    out_stride_b, out_stride_s, out_stride_dout,
    # Block sizes
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_DOUT: tl.constexpr,
):
    """
    Fused kernel for gate projection.
    
    The linear operation: output[b, d_out, s] = sum_k weight[d_out, k] * input[b, k, s] + bias[d_out]
    
    After transpose: transposed[b, s, d_out] = output[b, d_out, s]
    
    Final result: result[b, s, d_out] = transposed[b, s, d_out] * mul[b, s, d_out]
    
    Weight is stored as [D_in, D_out] in memory.
    For each output element [b, s, d_out], we compute:
      sum over k: input[b, k, s] * weight[k, d_out]
    where k ranges from 0 to D_in-1.
    """
    # Get program IDs for 2D grid (each thread handles multiple output elements)
    pid_b = tl.program_id(0)
    pid_s_start = tl.program_id(1) * BLOCK_SIZE_S
    pid_d_start = tl.program_id(2) * BLOCK_SIZE_DOUT
    
    # Get thread offsets within the block
    offs_s = pid_s_start + tl.arange(0, BLOCK_SIZE_S)
    offs_d = pid_d_start + tl.arange(0, BLOCK_SIZE_DOUT)
    
    # Mask for out-of-bounds
    mask_s = offs_s < S
    mask_d = offs_d < D_out
    mask_sd = mask_s[:, None] & mask_d[None, :]
    
    # Initialize output accumulators
    linear_acc = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_DOUT), dtype=tl.float32)
    
    # Compute the linear operation: sum over k
    for k in range(D_in):
        # Load input: input[b, k, s] -> shape [BLOCK_SIZE_S]
        input_offsets = pid_b * input_stride_b + k * input_stride_din + offs_s
        input_vals = tl.load(input_ptr + input_offsets, mask=mask_s, other=0.0)
        
        # Load weight: weight[k, d_out] -> shape [BLOCK_SIZE_DOUT]
        weight_offsets = k * weight_stride_din + offs_d
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask_d, other=0.0)
        
        # Accumulate: linear_acc[s, d] += input[k, s] * weight[k, d]
        # For proper broadcasting: input_vals[:, None] * weight_vals[None, :]
        linear_acc += input_vals[:, None] * weight_vals[None, :]
    
    # Load bias: bias[d_out] -> shape [BLOCK_SIZE_DOUT]
    bias_offsets = offs_d * bias_stride
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=mask_d, other=0.0)
    
    # Add bias
    linear_out = linear_acc + bias_vals[None, :]
    
    # Load mul: mul[b, s, d_out]
    mul_offsets = (pid_b * mul_stride_b + offs_s[:, None] * mul_stride_s + 
                   offs_d[None, :] * mul_stride_dout)
    mul_vals = tl.load(mul_ptr + mul_offsets, mask=mask_sd, other=0.0)
    
    # Compute final result
    result = linear_out * mul_vals
    
    # Store output
    output_offsets = (pid_b * out_stride_b + offs_s[:, None] * out_stride_s + 
                      offs_d[None, :] * out_stride_dout)
    tl.store(output_ptr + output_offsets, result, mask=mask_sd)


@torch.fx.wrap
def fused_gate_proj_wrapper(bias, weight, input, mul):
    """
    Fused gate projection: linear + transpose + multiply.
    
    Args:
        bias: [D_out] - bias for the linear layer
        weight: [D_in, D_out] - weight matrix (will be used as-is)
        input: [B, D_in, S] - input tensor
        mul: [B, S, D_out] - multiplication tensor
    
    Returns:
        output: [B, S, D_out]
    """
    B, D_in, S = input.shape
    D_out = mul.shape[2]
    
    # Allocate output
    output = torch.empty((B, S, D_out), dtype=input.dtype, device=input.device)
    
    # Calculate grid
    BLOCK_SIZE_S = 16
    BLOCK_SIZE_DOUT = 64
    grid = (
        B,
        triton.cdiv(S, BLOCK_SIZE_S),
        triton.cdiv(D_out, BLOCK_SIZE_DOUT),
    )
    
    # Launch kernel
    fused_gate_proj_kernel[grid](
        input, weight, bias, mul, output,
        B, S, D_in, D_out,
        input.stride(0), input.stride(1), input.stride(2),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        mul.stride(0), mul.stride(1), mul.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE_S,
        BLOCK_SIZE_DOUT,
    )
    
    return output


def pattern(bias, weight, input, mul):
    """
    Match the gate projection pattern:
    linear_out = linear(input, weight, bias)  # [B, D_out, S]
    transposed = linear_out.transpose(-1, -2)  # [B, S, D_out]
    result = mul * transposed  # [B, S, D_out]
    
    Returns result to maintain semantic equivalence (only the final output is returned).
    """
    linear_out = torch.nn.functional.linear(input, weight, bias)
    transposed = linear_out.transpose(-1, -2)
    result = mul * transposed
    return result


def replacement_args(bias, weight, input, mul):
    """
    Extract arguments needed for the fused kernel.
    """
    return (bias, weight, input, mul)


def replacement_func():
    """
    Returns the fused gate projection function.
    """
    return fused_gate_proj_wrapper