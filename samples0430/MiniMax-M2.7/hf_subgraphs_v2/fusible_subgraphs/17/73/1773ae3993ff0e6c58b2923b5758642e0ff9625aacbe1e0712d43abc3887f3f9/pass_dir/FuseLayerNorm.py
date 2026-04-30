import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_4):
    """
    Match the layer_norm operation for hidden_dim=384:
    tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
    
    Args:
        in_0: bias tensor [384]
        in_1: weight tensor [384]
        in_4: input tensor [batch, seq_len, 384]
    """
    return torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    """
    Extract arguments needed for the optimized layer_norm implementation.
    
    Returns:
        Tuple of (bias, weight, input, normalized_shape)
    """
    normalized_shape = (in_4.shape[-1],)
    return (in_0, in_1, in_4, normalized_shape)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_ptr,
    n_elements,
    normalized_shape: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused layer norm kernel with mean and rstd computation.
    
    Args:
        x_ptr: input tensor pointer
        weight_ptr: weight tensor pointer
        bias_ptr: bias tensor pointer
        mean_ptr: pointer to store mean (not used in output but needed for correctness)
        rstd_ptr: pointer to store reciprocal std (not used in output but needed for correctness)
        output_ptr: output tensor pointer
        n_elements: total number of elements in the normalized dimension
        normalized_shape: the shape to normalize over
        eps: epsilon for numerical stability
        BLOCK_SIZE: tilesize for computation
    """
    row_id = tl.program_id(0)
    seq_len = row_id
    
    # Compute the offset for this row
    row_offset = seq_len * n_elements
    
    # Load the row and compute sum for mean
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    tl.store(mean_ptr + row_id, mean)
    
    # Compute variance and rstd (reciprocal standard deviation)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row_id, rstd)
    
    # Normalize, scale, and shift
    x_norm = x_centered * rstd
    
    # Load weight and bias
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    
    # Apply weight and bias
    output = x_norm * w + b
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def triton_layer_norm_384(bias, weight, input_tensor, normalized_shape):
    """
    Optimized layer_norm for hidden_dim=384 using Triton kernel.
    
    Args:
        bias: bias tensor [normalized_shape]
        weight: weight tensor [normalized_shape]  
        input_tensor: input tensor [batch, seq_len, normalized_shape]
        normalized_shape: tuple specifying the normalized shape
    
    Returns:
        Normalized tensor with same shape as input
    """
    eps = 1e-12
    n_elements = normalized_shape[0]
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Allocate temporary buffers for mean and rstd
    mean = torch.empty((batch_size, seq_len), dtype=input_tensor.dtype, device=input_tensor.device)
    rstd = torch.empty((batch_size, seq_len), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid and block size
    BLOCK_SIZE = 1024  # Should be power of 2 and >= n_elements
    
    # Launch kernel: one block per row (batch * seq_len)
    num_programs = batch_size * seq_len
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        mean_ptr=mean,
        rstd_ptr=rstd,
        output_ptr=output,
        n_elements=n_elements,
        normalized_shape=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """Return the optimized layer_norm function."""
    return triton_layer_norm_384