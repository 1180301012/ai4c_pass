import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused layer norm kernel with mean and rstd computation.
    
    Args:
        x_ptr: input tensor pointer
        weight_ptr: weight tensor pointer
        bias_ptr: bias tensor pointer
        mean_ptr: pointer to store mean
        rstd_ptr: pointer to store reciprocal std
        output_ptr: output tensor pointer
        n_elements: total number of elements in the normalized dimension
        eps: epsilon for numerical stability
        BLOCK_SIZE: tilesize for computation
    """
    row_id = tl.program_id(0)
    
    # Compute the offset for this row
    row_offset = row_id * n_elements
    
    # Load the row
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using reduction
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
def triton_layer_norm_dispatch(bias, weight, input_tensor, normalized_shape, route=""):
    """
    Dispatch layer_norm based on route string.
    
    Args:
        bias: bias tensor
        weight: weight tensor  
        input_tensor: input tensor
        normalized_shape: tuple specifying the normalized shape
        route: routing string for different hidden dims (e.g., "384", "768", "32")
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
    
    # Auto-tune block size based on n_elements
    # For small hidden dimensions, use smaller block sizes for better efficiency
    if n_elements <= 32:
        BLOCK_SIZE = 64
    elif n_elements <= 128:
        BLOCK_SIZE = 256
    elif n_elements <= 384:
        BLOCK_SIZE = 512
    elif n_elements <= 768:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
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
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output