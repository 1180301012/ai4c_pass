import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_dropout_transpose_kernel_005(
    input_ptr, weight_ptr, bias_ptr,
    output_dropout_ptr, output_transpose_ptr,
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    seed: tl.constexpr,
):
    """
    Fused kernel for: linear + dropout(0.05) + transpose
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
    
    # Convert to output dtype
    result = acc.to(tl.float32)  # Use float32 for precision
    
    # Apply dropout with p=0.05
    rng_offset = seed + pid_b * 1000 + pid_d
    random = tl.rand(rng_offset, 0.0)
    mask = random > 0.05
    result = tl.where(mask, result / 0.95, 0.0)
    
    # Store to both outputs
    for s in range(S):
        dropout_offset = pid_b * S * D_out + s * D_out + pid_d
        transpose_offset = pid_b * D_out * S + pid_d * S + s
        
        result_i16 = result.to(tl.float32)  # Keep precision for bfloat16
        tl.store(output_dropout_ptr + dropout_offset, result_i16)
        tl.store(output_transpose_ptr + transpose_offset, result_i16)


@torch.fx.wrap
def fused_linear_dropout_transpose_wrapper_005(in_0, in_1, in_2):
    """
    Wrapper for the fused kernel with dropout p=0.05.
    Returns: (dropout_output [B, S, D_out], transpose_output [B, D_out, S])
    """
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    # Create output tensors with proper dtype
    output_dropout = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    output_transpose = torch.empty(B, D_out, S, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration
    grid = (B, D_out)
    
    BLOCK_SIZE_K = min(32, D_in)
    seed = 12345
    
    fused_linear_dropout_transpose_kernel_005[grid](
        in_2, in_1, in_0,
        output_dropout, output_transpose,
        B, S, D_in, D_out,
        BLOCK_SIZE_K,
        seed,
    )
    
    return output_dropout, output_transpose


def pattern(in_0, in_1, in_2):
    """Match: linear + dropout(0.05) + transpose -> (tmp_3, tmp_4)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_dropout_transpose_wrapper_005