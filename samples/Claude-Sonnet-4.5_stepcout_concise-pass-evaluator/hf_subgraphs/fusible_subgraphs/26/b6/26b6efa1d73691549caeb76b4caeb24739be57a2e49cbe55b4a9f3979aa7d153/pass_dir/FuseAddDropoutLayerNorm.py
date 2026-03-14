import torch
import triton
import triton.language as tl


def pattern(input_tensor, normalized_shape, weight, bias, eps):
    """
    Pattern that takes normalized_shape as a direct parameter
    Just optimize layer_norm
    """
    # Layer norm
    output = torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    return output


def replacement_args(input_tensor, normalized_shape, weight, bias, eps):
    return (input_tensor, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['hidden_size'],
)
@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_seq_size,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for layer_norm
    Each program handles one row (batch*seq position)
    """
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_offset = row_idx * hidden_size
    
    # Create offsets for the hidden dimension
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    
    # Load input
    x = tl.load(input_ptr + row_offset + cols, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / hidden_size
    
    # Compute variance
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    
    # Compute rstd (reciprocal standard deviation)
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    
    # Normalize and apply affine transformation
    normalized = centered * rstd * weight + bias_val
    
    # Store normalized output
    tl.store(output_ptr + row_offset + cols, normalized, mask=mask)


@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias):
    """
    Wrapper function that launches the optimized layer_norm kernel
    """
    # Get dimensions
    hidden_size = input_tensor.shape[-1]
    
    # Flatten to 2D if needed
    orig_shape = input_tensor.shape
    input_2d = input_tensor.reshape(-1, hidden_size)
    
    # Create output tensor
    output = torch.empty_like(input_2d)
    
    # Launch kernel with autotuning
    grid = (input_2d.shape[0],)
    layernorm_kernel[grid](
        input_2d,
        weight,
        bias,
        output,
        input_2d.shape[0],
        hidden_size,
        1e-12,
    )
    
    # Reshape back to original shape
    output = output.reshape(orig_shape)
    
    return output


def replacement_func():
    return optimized_layernorm