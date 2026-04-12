import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching: linear + transpose + element-wise multiply
    This matches the computation: output = in_3 * (in_2 @ in_1.T + in_0).T
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    transpose = linear.transpose(-1, -2)
    result = in_3 * transpose
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_transpose_multiply_kernel(
    input_ptr,      # Input tensor [batch, in_features, out_features], stride=in_features*out_features
    weight_ptr,     # Weight tensor [out_features, in_features], stride=in_features
    bias_ptr,       # Bias tensor [out_features]
    mul_ptr,        # Multiplication tensor [batch, out_features, in_features]
    output_ptr,     # Output tensor [batch, out_features, in_features]
    batch_size,
    in_features,
    out_features,
    BLOCK_K: tl.constexpr
):
    """
    Fused kernel: output = mul * (input @ weight.T + bias).T
    Each program computes one output element [b, o, i]
    """
    # Program IDs for 2D grid (batch x out_features)
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    # Compute input and output offsets
    input_offset_base = pid_b * in_features * out_features + pid_o * in_features
    output_offset = pid_b * out_features * in_features + pid_o * in_features
    
    # Create offset arrays for K dimension
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < min(BLOCK_K, in_features)
    
    # Initialize accumulator with bias
    acc = tl.load(bias_ptr + pid_o, mask=pid_o < out_features, other=0.0).to(tl.float32)
    
    # Matrix multiplication: element (b, o) = sum_k input[b, o, k] * weight[o, k]
    for k in range(0, in_features, BLOCK_K):
        k_end = min(k + BLOCK_K, in_features)
        
        # Load input values: input[b, o, k_start+k]
        input_ptrs = input_ptr + input_offset_base + k_offs
        input_vals = tl.load(
            input_ptrs,
            mask=k_offs < k_end - k,
            other=0.0
        ).to(tl.float32)
        
        # Load weight values: weight[o, k_start+k]
        weight_ptrs = weight_ptr + pid_o * in_features + k_offs
        weight_vals = tl.load(
            weight_ptrs, 
            mask=k_offs < k_end - k,
            other=0.0
        ).to(tl.float32)
        
        # Matrix multiplication
        acc += tl.sum(input_vals * weight_vals)
    
    # Load multiplier value: mul[b, o, :] - all for this (b,o) pair
    mul_offset = (pid_b * out_features + pid_o) * in_features
    mul_vals = tl.load(
        mul_ptr + mul_offset + k_offs,
        mask=k_mask,
        other=1.0
    ).to(tl.float32)
    
    # Compute final result: transpose_result * mul
    final_result = acc.to(output_ptr.type.element_ty) * mul_vals
    
    # Store result for all input dimensions
    tl.store(output_ptr + output_offset + k_offs, final_result, mask=k_mask)

@torch.fx.wrap
def fused_linear_transpose_multiply(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused Triton kernel
    """
    batch_size, in_features, out_features = in_2.shape
    
    # Output shape should be [batch_size, out_features, in_features]
    output = torch.empty((batch_size, out_features, in_features), dtype=in_2.dtype, device=in_2.device)
    
    # Calculate grid dimensions - one program per (batch, out_features) pair
    # Each program computes a full vector [for all input_features] for one (batch, out_features) pair
    num_batch_blocks = (batch_size + 127) // 128
    num_out_blocks = (out_features + 127) // 128
    
    fused_linear_transpose_multiply_kernel[(num_batch_blocks, num_out_blocks)](
        input_ptr=in_2,
        weight_ptr=in_1, 
        bias_ptr=in_0,
        mul_ptr=in_3,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        BLOCK_K=128
    )
    
    return output

def replacement_func():
    """Returns the fused kernel function"""
    return fused_linear_transpose_multiply