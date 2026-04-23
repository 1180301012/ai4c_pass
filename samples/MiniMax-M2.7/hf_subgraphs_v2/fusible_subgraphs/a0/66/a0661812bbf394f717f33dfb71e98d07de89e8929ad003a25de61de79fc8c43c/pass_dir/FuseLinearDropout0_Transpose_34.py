import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_transpose_kernel_0(
    input_ptr, weight_ptr, bias_ptr,
    output_dropout_ptr, output_transpose_ptr,
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: linear + (dropout with p=0.0, no-op) + transpose
    Input: [B, S, D_in]
    Weight: [D_out, D_in] 
    Bias: [D_out]
    Output (dropout): [B, S, D_out]
    Output (transpose): [B, D_out, S]
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Pre-compute offsets for this output feature
    weight_offsets = pid_d * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    # Accumulator for matmul
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over sequences
    for s in range(S):
        # Load weight row for this output feature
        w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        
        # Load input vector for this batch and sequence
        input_offsets = pid_b * S * D_in + s * D_in + tl.arange(0, BLOCK_SIZE_K)
        input_mask = input_offsets < B * S * D_in
        inp = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Matmul accumulation
        acc += tl.sum(inp * w)
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_d)
        acc = acc + bias
    
    # No dropout needed for p=0.0 during inference
    
    # Store to both outputs (same result for both)
    for s in range(S):
        dropout_offset = pid_b * S * D_out + s * D_out + pid_d
        transpose_offset = pid_b * D_out * S + pid_d * S + s
        
        # Convert to output dtype
        result = acc.to(output_dropout_ptr.dtype)
        tl.store(output_dropout_ptr + dropout_offset, result)
        tl.store(output_transpose_ptr + transpose_offset, result)


@torch.fx.wrap
def fused_linear_transpose_wrapper_0_34(in_0, in_1, in_2):
    """
    Wrapper for fused linear + transpose (dropout p=0.0 is no-op).
    Returns: (dropout_output [B, S, D_out], transpose_output [B, D_out, S])
    """
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    # Create output tensors
    output_dropout = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    output_transpose = torch.empty(B, D_out, S, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration
    grid = (B, D_out)
    
    BLOCK_SIZE_K = min(32, D_in)
    
    fused_linear_transpose_kernel_0[grid](
        in_2, in_1, in_0,
        output_dropout, output_transpose,
        B, S, D_in, D_out,
        BLOCK_SIZE_K,
    )
    
    return output_dropout, output_transpose


def pattern(in_0, in_1, in_2):
    """Match: linear + dropout(0.0) + transpose -> (tmp_3, tmp_4)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_transpose_wrapper_0_34