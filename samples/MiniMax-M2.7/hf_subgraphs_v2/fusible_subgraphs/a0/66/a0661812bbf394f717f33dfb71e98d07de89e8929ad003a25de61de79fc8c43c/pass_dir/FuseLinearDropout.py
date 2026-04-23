import torch
import triton
import triton.language as tl

# Shared Triton kernel for fused linear + dropout + transpose
@triton.jit
def fused_linear_dropout_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_dropout_ptr, output_transpose_ptr,
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    seed: tl.constexpr,
):
    """
    Fused kernel for: linear + dropout + transpose
    Input: [B, S, D_in], Weight: [D_out, D_in], Bias: [D_out]
    Output (dropout): [B, S, D_out]
    Output (transpose): [B, D_out, S]
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Compute offsets for weight row
    weight_offsets = pid_d * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    # Accumulator for matmul
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over sequences - each program handles one output feature for one batch
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
    
    # Apply dropout (for p=0.0, all elements are kept)
    if dropout_p > 0.0:
        rng_offset = seed + pid_b * 1000 + pid_d
        random = tl.rand(rng_offset, 0.0)
        mask = random > dropout_p
        scale = 1.0 / (1.0 - dropout_p)
        result = tl.where(mask, acc * scale, 0.0)
    else:
        result = acc
    
    # Store to dropout output [B, S, D_out]
    for s in range(S):
        dropout_offset = pid_b * S * D_out + s * D_out + pid_d
        transpose_offset = pid_b * D_out * S + pid_d * S + s
        
        result_dtype = result.to(tl.float16)  # Convert to output dtype
        tl.store(output_dropout_ptr + dropout_offset, result_dtype)
        tl.store(output_transpose_ptr + transpose_offset, result_dtype)


@torch.fx.wrap
def fused_linear_dropout_transpose_wrapper(in_0, in_1, in_2, dropout_p, return_order):
    """
    Wrapper for the fused kernel.
    in_0: bias tensor [D_out]
    in_1: weight tensor [D_out, D_in]
    in_2: input tensor [B, S, D_in]
    dropout_p: dropout probability
    return_order: 0 for (dropout, transpose), 1 for (transpose, dropout)
    Returns: tuple of (dropout_output, transpose_output)
    """
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    # Create output tensors
    output_dropout = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    output_transpose = torch.empty(B, D_out, S, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration: one program per (batch, output_feature)
    grid = (B, D_out)
    
    # Kernel parameters
    BLOCK_SIZE_K = min(32, D_in)
    seed = 12345
    
    fused_linear_dropout_transpose_kernel[grid](
        in_2, in_1, in_0,
        output_dropout, output_transpose,
        B, S, D_in, D_out,
        dropout_p,
        BLOCK_SIZE_K,
        seed,
    )
    
    # Return based on order
    if return_order == 0:
        return output_dropout, output_transpose
    else:
        return output_transpose, output_dropout


# Pattern 1: dropout(0.1) + return (tmp_3, tmp_4)
def pattern_p01(in_0, in_1, in_2):
    """Match: linear + dropout(0.1) + transpose -> (tmp_3, tmp_4)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p01(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 0.1, 0)


def replacement_func_p01():
    return fused_linear_dropout_transpose_wrapper


# Pattern 2: dropout(0.05) + return (tmp_3, tmp_4)
def pattern_p005(in_0, in_1, in_2):
    """Match: linear + dropout(0.05) + transpose -> (tmp_3, tmp_4)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p005(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 0.05, 0)


def replacement_func_p005():
    return fused_linear_dropout_transpose_wrapper


# Pattern 3: dropout(0.0) + return (tmp_3, tmp_4)
def pattern_p0_34(in_0, in_1, in_2):
    """Match: linear + dropout(0.0) + transpose -> (tmp_3, tmp_4)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p0_34(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 0.0, 0)


def replacement_func_p0_34():
    return fused_linear_dropout_transpose_wrapper


# Pattern 4: dropout(0.0) + return (tmp_4, tmp_3)
def pattern_p0_43(in_0, in_1, in_2):
    """Match: linear + dropout(0.0) + transpose -> (tmp_4, tmp_3)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4, tmp_3


def replacement_args_p0_43(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 0.0, 1)


def replacement_func_p0_43():
    return fused_linear_dropout_transpose_wrapper