import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, training, momentum, eps):
    """
    Pattern to match batch_norm operation in inference mode
    """
    output = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    return output

def replacement_args(input, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input, running_mean, running_var, weight, bias, eps)

@triton.jit
def fused_batchnorm_kernel(
    input_ptr, output_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    eps,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused batch normalization kernel for inference (2D input: [batch, channels])
    output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
    """
    row_idx = tl.program_id(0)
    
    # Offset for this row
    row_start = row_idx * N
    
    # Process the entire row in chunks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        block_offsets = block_start + col_offsets
        mask = block_offsets < N
        
        # Load input
        input_vals = tl.load(input_ptr + row_start + block_offsets, mask=mask, other=0.0)
        
        # Load running statistics and affine parameters
        mean = tl.load(running_mean_ptr + block_offsets, mask=mask, other=0.0)
        var = tl.load(running_var_ptr + block_offsets, mask=mask, other=0.0)
        gamma = tl.load(weight_ptr + block_offsets, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + block_offsets, mask=mask, other=0.0)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        inv_std = tl.rsqrt(var + eps)
        normalized = (input_vals - mean) * inv_std
        
        # Apply affine transformation: gamma * normalized + beta
        output_vals = gamma * normalized + beta
        
        # Store output
        tl.store(output_ptr + row_start + block_offsets, output_vals, mask=mask)


@torch.fx.wrap
def fused_batchnorm_wrapper(input, running_mean, running_var, weight, bias, eps):
    """
    Wrapper for fused batch normalization
    """
    # Get dimensions
    if input.dim() == 2:
        M, N = input.shape
    else:
        raise ValueError(f"Expected 2D input, got {input.dim()}D")
    
    output = torch.empty_like(input)
    
    # Choose block size (384 is close to actual N=384)
    BLOCK_SIZE = 256
    
    grid = (M,)
    
    fused_batchnorm_kernel[grid](
        input, output,
        running_mean, running_var,
        weight, bias,
        eps,
        N,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_batchnorm_wrapper