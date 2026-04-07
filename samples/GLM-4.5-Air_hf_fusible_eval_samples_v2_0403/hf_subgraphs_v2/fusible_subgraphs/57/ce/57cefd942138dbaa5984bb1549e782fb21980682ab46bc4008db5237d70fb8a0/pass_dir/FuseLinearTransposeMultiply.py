import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the computation pattern: linear + transpose + multiply
    """
    # Linear transformation: output = in_2 @ in_1.T + in_0
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    # Transpose the last two dimensions
    transpose_result = linear.transpose(-1, -2)
    # Element-wise multiplication with in_3
    result = in_3 * transpose_result
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the fused kernel
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def linear_transpose_multiply_kernel(
    bias_ptr,          # [H_out] - bias vector
    weight_ptr,        # [H_out, H_out] - weight matrix  
    input_ptr,         # [B, H_in, H_out] - input tensor
    multi_ptr,         # [B, H_out, H_in] - multiplication tensor
    output_ptr,        # [B, H_out, H_in] - output tensor
    B, H_in, H_out,   # Tensor dimensions
):
    """
    Fused kernel that combines linear transformation + transpose + element-wise multiply
    This kernel computes: output[i,j] = multi[i,j] * (sum_k input[i,k] * weight[j,k] + bias[j])
    """
    # Program ID determines which output element this thread computes
    batch_id = tl.program_id(0)
    h_out_idx = tl.program_id(1) 
    h_in_idx = tl.program_id(2)
    
    # Skip if out of bounds
    if h_out_idx >= H_out or h_in_idx >= H_in:
        return
    
    # Load bias for this output dimension
    bias_val = tl.load(bias_ptr + h_out_idx)
    
    # Compute dot product: sum over k dimension (H_in)
    weighted_sum = 0.0
    for k in range(0, H_in):
        # Load input and weight elements
        input_val = tl.load(input_ptr + batch_id * H_in * H_out + k * H_out + h_out_idx)
        weight_val = tl.load(weight_ptr + h_out_idx * H_out + k)
        weighted_sum += input_val * weight_val
    
    # Add bias and multiply by multi element
    linear_result = weighted_sum + bias_val
    multi_val = tl.load(multi_ptr + batch_id * H_out * H_in + h_out_idx * H_in + h_in_idx)
    
    # Store final result
    output_val = linear_result * multi_val
    tl.store(output_ptr + batch_id * H_out * H_in + h_out_idx * H_in + h_in_idx, output_val)

@torch.fx.wrap
def fused_linear_transpose_multiply(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused kernel
    """
    # Get tensor shapes
    B, H_in, H_out = in_2.shape
    
    # Set up grid dimensions
    # Each thread computes one output element
    grid = (
        B,                          # Batch dimension  
        (H_out + 31) // 32,         # H_out dimension (warps)
        (H_in + 31) // 32,          # H_in dimension (warps)
    )
    
    # Create output tensor
    output = torch.empty_like(in_3)
    
    # Launch the kernel
    linear_transpose_multiply_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        multi_ptr=in_3,
        output_ptr=output,
        B=B,
        H_in=H_in,
        H_out=H_out,
    )
    
    return output

def replacement_func():
    """
    Returns the fused kernel function
    """
    return fused_linear_transpose_multiply